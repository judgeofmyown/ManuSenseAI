import numpy as np
import torch
from torch import Tensor

def freq_matrix(x: Tensor, N: int, T: int) -> Tensor:
    """
    Parameters:
    - x: Tensor of shape [L], the raw time series
    - N: int, window size for first stack
    - T: int, window size for second stack (row-wise window)

    Returns:
    - Tensor of shape [M, H, F] where:
      M = L - N + 1
      H = N - T + 1
      F = T // 2 (for real FFT)
    """

    # [x1, x2, x3, x4, .. xL]
    assert x.dim() == 1
    L = len(x) 
    M = L - N + 1
    H = N - T + 1
    F = T//2

    # first stack
    time_series_dataset = x.unfold(dimension=0, size=N, step=1)
    
    freq_matrices = []

    for i in range(M):
        window = time_series_dataset[i]

        ts_matrix = window.unfold(dimension=0, size=T, step=1)

        ts_matrix = ts_matrix.to(torch.complex64)

        fft_matrix = torch.fft.fft(ts_matrix, dim=1)

        fft_matrix = fft_matrix[: , :F]

        freq_matrices.append(freq_matrix)

    return torch.stack(freq_matrices, dim=0)



