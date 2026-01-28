import math
import torch

def execute(steps = 20, max_shift = 2.05, base_shift = 0.95, stretch = True, terminal = 0.1):
    tokens = 4096

    sigmas = torch.linspace(1.0, 0.0, steps + 1)

    x1 = 1024
    x2 = 4096
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    sigma_shift = (tokens) * mm + b

    power = 1
    sigmas = torch.where(
        sigmas != 0,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
        0,
    )

    # Stretch sigmas so that its final value matches the given terminal value.
    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas[non_zero_mask] = stretched

    return sigmas

if __name__ == "__main__":
    print(execute())