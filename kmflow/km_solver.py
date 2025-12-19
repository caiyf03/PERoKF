
import os
import math
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from dataclasses import dataclass, field

from .GaussianRF import GaussianRF
from .km_config import KMFlow2D_Config

class KMFlow2D_Solver(object):
    def __init__(self, w0, Re, n):

        # Grid size

        self.s = w0.size()[-1]

        assert self.s == w0.size()[-2], "Grid must be uniform in both directions."

        assert math.log2(self.s).is_integer(), "Grid size must be power of 2."

        assert n >= 0 and isinstance(n, int), "Forcing number must be non-negative integer."

        assert n < self.s // 2 - 1, "Forcing number too large for grid size."

        # Forcing number
        self.n = n

        assert Re > 0, "Reynolds number must be positive."

        # Reynolds number
        self.Re = Re

        # Device
        self.device = w0.device

        # Current time
        self.time = 0.0

        # Current vorticity in Fourier space
        self.w_h = torch.fft.fft2(w0, norm="backward")

        # Wavenumbers in y and x directions
        self.k_y = torch.cat((torch.arange(start=0, end=self.s // 2, step=1, dtype=torch.float32, device=self.device), \
                              torch.arange(start=-self.s // 2, end=0, step=1, dtype=torch.float32, device=self.device)),
                             0).repeat(self.s, 1)

        self.k_x = self.k_y.clone().transpose(0, 1)

        # Negative inverse Laplacian in Fourier space
        self.inv_lap = (self.k_x ** 2 + self.k_y ** 2)
        self.inv_lap[0, 0] = 1.0
        self.inv_lap = 1.0 / self.inv_lap

        # Negative scaled Laplacian
        self.G = (1.0 / self.Re) * (self.k_x ** 2 + self.k_y ** 2)
        self.linear_term = self.G + 0.1   # drag

        # Dealiasing mask using 2/3 rule
        self.dealias = (self.k_x ** 2 + self.k_y ** 2 <= (self.s / 3.0) ** 2).float()
        # Ensure mean zero
        self.dealias[0, 0] = 0.0

    # Get current vorticity from stream function (Fourier space)
    def vorticity(self, stream_f=None, real_space=True):
        if stream_f is not None:
            w_h = self.Re * self.G * stream_f
        else:
            w_h = self.w_h

        if real_space:
            return torch.fft.irfft2(w_h, s=(self.s, self.s), norm="backward")
        else:
            return w_h

    # Compute stream function from vorticity (Fourier space)
    def stream_function(self, w_h=None, real_space=False):
        if w_h is None:
            psi_h = self.w_h.clone()
        else:
            psi_h = w_h.clone()

        # Stream function in Fourier space: solve Poisson equation
        psi_h = self.inv_lap * psi_h

        if real_space:
            return torch.fft.irfft2(psi_h, s=(self.s, self.s), norm="backward")
        else:
            return psi_h

    # Compute velocity field from stream function (Fourier space)
    def velocity_field(self, stream_f=None, real_space=True):
        if stream_f is None:
            stream_f = self.stream_function(real_space=False)

        # Velocity field in x-direction = psi_y
        q_h = stream_f * 1j * self.k_y

        # Velocity field in y-direction = -psi_x
        v_h = stream_f * -1j * self.k_x

        if real_space:
            q = torch.fft.irfft2(q_h, s=(self.s, self.s), norm="backward")
            v = torch.fft.irfft2(v_h, s=(self.s, self.s), norm="backward")
            return q, v
        else:
            return q_h, v_h

    # Compute non-linear term + forcing from given vorticity (Fourier space)
    def explicit_term(self, w_h):
        # Physical space vorticity
        w = torch.fft.ifft2(w_h, s=(self.s, self.s), norm="backward")

        # Velocity field in physical space
        q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)

        # Compute non-linear term
        t1 = torch.fft.fft2(q * w, s=(self.s, self.s), norm="backward")
        t1 = self.k_x * t1

        t2 = torch.fft.fft2(v * w, s=(self.s, self.s), norm="backward")
        t2 = self.k_y * t2

        nonlin = -1j * (t1 + t2)

        if self.n > 0:
            nonlin[..., 0, self.n] -= (float(self.n) / 2.0) * (self.s ** 2)
            nonlin[..., 0, -self.n] -= (float(self.n) / 2.0) * (self.s ** 2)

        return nonlin

    def implicit_term(self, w_h):
        return self.linear_term * w_h

    def implicit_solve(self, w_h, time_step):
        return 1 / (1 + time_step * self.linear_term) * w_h

    def advance(self, t, delta_t=1e-3):
        # Final time
        T = self.time + t

        # Advance solution in Fourier space
        while self.time < T:

            if self.time + delta_t > T:
                current_delta_t = T - self.time
            else:
                current_delta_t = delta_t

            # Inner-step of Heun's method

            g = self.w_h - 0.5 * current_delta_t * self.implicit_term(self.w_h)
            h1 = self.explicit_term(self.w_h)
            w_h_tilde = self.implicit_solve(g + current_delta_t*h1, 0.5*current_delta_t)

            # Cranck-Nicholson + Heun update

            h2 = 0.5 * (self.explicit_term(w_h_tilde) + h1)
            self.w_h = self.implicit_solve(g + current_delta_t*h2, 0.5*current_delta_t)
            # De-alias
            self.w_h *= self.dealias
            self.time += current_delta_t

            if torch.any(torch.isnan(self.w_h)):
                raise Exception('NaN in voriticlity field')
            
    def deriv_vorticity(self, real_space=True):
        dw_h_dt = self.explicit_term(self.w_h) - self.implicit_term(self.w_h)
        if real_space:
            return torch.fft.irfft2(dw_h_dt, s=(self.s, self.s), norm="backward")
        else:
            return dw_h_dt

class KMFlow2D_Generator(object):
    def __init__(self, config: KMFlow2D_Config):
        self.config = config

        self.Re = config.Re
        self.forcing_num = config.forcing_num
        self.size = config.size
        self.downsample = config.downsample
        self.spin_up = config.spin_up
        self.duration = int(config.duration)
        self.fps = config.fps
        self.dt = 1.0 / config.fps
        self.batch_size = config.batch_size
        self.device = config.device

        self.root_dir = config.root_dir

        self.save_vorticity = config.save_vorticity
        self.save_velocity = config.save_velocity

        self.samples = config.samples
        self.start_seed = config.start_seed

        self.output_size = self.size // self.downsample

        print(f"Reynolds numbers: {self.Re}")

    # Saved data shape: [channel, height, width]
    def _generate_impl(self, seed, Re, forcing_num):
        np.random.seed(seed*666)
        GRF = GaussianRF(2, self.size, 2 * math.pi, alpha=2.5, tau=7, device=self.device)
        u0 = GRF.sample(self.batch_size)

        NS = KMFlow2D_Solver(u0, Re, forcing_num)
        NS.advance(self.spin_up, delta_t=1e-4)

        for i in tqdm(range(self.duration)):
            for j in range(self.fps):
                NS.advance(self.dt, delta_t=1e-4)

                sol = NS.vorticity().cpu().numpy()

                for b in range(sol.shape[0]):
                    dir = os.path.join(self.root_dir, f'size{self.size}/kf_2d_re{int(Re)}/fn{int(forcing_num)}/seed{seed*self.batch_size+b}')
                    os.makedirs(dir, exist_ok=True)
                    
                    if self.save_vorticity:
                        if not os.path.exists(os.path.join(dir, 'vorticity')):
                            os.makedirs(os.path.join(dir, 'vorticity'))

                        np.save(
                            os.path.join(dir, 'vorticity', f'sol_t{i}_step{j}'), 
                            sol[b, None, ::self.downsample, ::self.downsample]
                        )

                    if self.save_velocity:
                        q, v = NS.velocity_field(real_space=True)
                        q = q.cpu().numpy()
                        v = v.cpu().numpy()

                        if not os.path.exists(os.path.join(dir, 'velocity')):
                            os.makedirs(os.path.join(dir, 'velocity'))

                        np.save(
                            os.path.join(dir, 'velocity', f'sol_t{i}_step{j}'), 
                            np.concat(
                                (
                                    q[b, None, ::self.downsample, ::self.downsample], 
                                    v[b, None, ::self.downsample, ::self.downsample]
                                ), axis=0
                            )
                        )

    # There will be in total {batch size * samples} for each Re and forcing number
    # In each file, the data shape is [channel, height, width] = [channel, size // downsample, size // downsample]
    # channel = 1 for vorticity, channel = 2 for velocity field (u,v)
    def generate(self):
        for Re in self.Re:
            for forcing_num in self.forcing_num:
                for i in range(self.samples):
                    self._generate_impl(seed=i + self.start_seed, Re=Re, forcing_num=forcing_num)


if __name__ == "__main__":

    def main():
        # Example configuration
        # data amount: len(Re) * len(forcing_num) * samples * batch_size * duration * fps
        config = KMFlow2D_Config(
            size = 256,
            downsample = 1,
            Re = [1000.0, 2000.0, 3000.0],
            forcing_num = [8, 12, 16],
            spin_up = 1.0,
            duration = 5.0,
            samples = 2,
            save_vorticity = True,
            batch_size = 5,
        )

        gen = KMFlow2D_Generator(config)

        gen.generate()

    main()