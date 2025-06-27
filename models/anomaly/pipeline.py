import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from anomaly.pipeline import feature_freq_domain

"""
    FSENet: Feature Squeeze-and-Excitation Network for Frequency Matrix Attention
    ---------------------------------------------------------------------------

    This module implements the core attention mechanism used in the F-SENet-LSTM architecture.
    It enhances the representation power of frequency-transformed time series data by applying
    both local convolution over the frequency domain and global channel-wise excitation.

    Input Shape:
        - Tensor of shape (C, H, F+1)
            C  : Number of input channels (e.g., features, sensors)
            H  : Number of sub-time-windows (rows of the frequency matrix)
            F+1: Number of frequency bins (e.g., from FFT of windowed signals)

    Processing Steps:
        1. **Convolution Over Frequency Domain**:
            Each row in the input matrix represents a frequency spectrum. A 1D convolution is
            applied along the frequency axis (F+1) to capture local relationships between neighboring
            frequency bins. This produces an intermediate tensor:
            
                U ∈ ℝ^{H × W × C'}
            
            where:
                - W is the new frequency length (post-convolution)
                - C' is the number of convolution filters (channels)

        2. **Global Average Pooling**:
            For each feature map (conv output channel), global average pooling is performed over
            the H × W dimensions to obtain a compact descriptor:
            
                P_c = (1 / (H × W)) × ∑_h ∑_w U_c(h, w)
                
            These descriptors are stacked into a vector:
                
                P ∈ ℝ^{1 × 1 × C'}

        3. **Channel-Wise Excitation**:
            The pooled descriptor vector P is passed through a lightweight fully connected network
            (bottleneck architecture) to learn the importance of each channel. It outputs attention weights:

                z = σ(W₂(ReLU(W₁(P))))
                
            where σ is the sigmoid activation, and W₁, W₂ are learned linear layers.

        4. **Recalibration and Activation**:
            The learned weights z are used to scale the original feature maps U via channel-wise
            multiplication. Optionally, a `tanh` activation can be applied to improve nonlinearity:

                V_c(h, w) = tanh(z_c × U_c(h, w))

    Output:
        - Tensor of shape (H, W, C') — attention-enhanced frequency representation,
          ready to be passed into downstream modules (e.g., LSTM, MLP).

    Advantages:
        - Captures both local frequency structure and global channel dependencies.
        - Lightweight and modular attention mechanism.
        - Adaptable for multivariate time series or single-channel frequency maps.

    Example:
        >>> fsenet = FSENet(in_channels=1, conv_out_channels=32, reduction=16)
        >>> x = torch.randn(1, 64, 33)  # [C=1, H=64, F+1=33]
        >>> out = fsenet(x)
        >>> print(out.shape)  # [64, W, 32]

    References:
        - F-SENet-LSTM: "A Hybrid Deep Learning Model Based on Squeeze-and-Excitation Networks and LSTM for Time Series Classification"
"""
class FSENet(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int):
        super(FSENet, self).__init__()
        self.out_channels = out_channels 
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size)
    
    # Can normalize the frequecy matric before feed to convolution layer [Tip/ToBeNoted]
    
    def forward(self, x: Tensor):
        # x: shape [C, H, F+1]
        C = x.shape[0]
        H = x.shape[1]
        F = x.shape[2] - 1
        
        result = []

        for i in range(C):
            # extracting one channels frequency matrix [H, F+1]
            freq_matrix = x[i]
            
            # Reshape for Conv1d [H, 1, F+1]
            freq_matrix = freq_matrix.view(H, 1, F+1)

            # Conv1d operated on dim=2,  across frequency
            freq_conv = self.conv(freq_matrix) # shape -> [H, C', W]
            
            # Permute th [C', H, W]
            freq_conv = freq_conv.permute(1, 0, 2)
            
            # Global average pooling over spatial dims (H, W) per out channel
            weights = freq_conv.mean(dim=(1, 2)) # shape: [C']
            
            # Reshape for broadcastig: [C', 1, 1]
            weights = weights.view(-1, 1, 1)
            
            # element wise (Hadamard) multiplication
            reweighted = freq_conv * weights
            
            # Apply non-linearity
            V_hw = torch.tanh(reweighted) # shape: [C', H, W]

            results.append(V_hw)
        
        # stack over original channels -> shape: [C, C', H, W]
        return torch.stack(results, dim=0), V_hw.shape[1], V_hw.shape[3]

# Good Ol simple LSTM model 
"""
LSTM(Long short term memory):
    -designed to capture temporal dependencies in sequence, while mitigating vanishing gradients using gating mechanisms
    -Each time step: 
        x: input vector: R{input_size}
        maintains:
            h_t: hidden_states: R{hidden_size}
            c_t: cell_state: R{hiddden_size}
"""

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=False) 
        

    def forward(self, x):
        # input x: [C, C', H, W]
        C = x.shape[0]
        C_conv_channels = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        
        results = []

        for i in range(C):
            # extract the single segment required
            x_segment= x[i] # shape: [C', H, W]
            
            x_segment = x_segment.permute(1, 0, 2) # shape: [H, C', W]
            
            # input for LSTM
            x_segment = x_segment.reshape(H, -1).unsqueeze(1) # shape: [H, 1, C'*W]
            
            # LSTM expects [seq_len, batch_size, input_size]
            output, (hn, cn) = self.lstm(x_segment)

            # collect final hidden state for last layer
            results.append(hn[-1]) # shape: [hidden_size]

        # stack all outputs: shape: [C, hidden_size]
        return torch.stack(results, dim=0)

class DNN(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        
        layers = []
        in_dim = input_size

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = h

        layers.append(nn.Linear(in_dim, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) # x shape: [C, input_size]

