"""
cnn.py – DV-STTGAT  (NYU-tuned)
=================================
Multi-Scale Inception CNN for temporal feature extraction from BOLD signals.

NYU-specific tuning (TR=2000ms):
    Kernel sizes changed from (3, 7, 11) → (3, 5, 7).
    Smaller kernels capture fast BOLD dynamics that are consistent across
    NYU's homogeneous scanner environment, without over-smoothing.

    SOTA Update: Temporal Max Pooling (k=2, s=2) is applied after each branch
    BEFORE concatenation. This captures the most intense "bursts" of activity
    and halves the temporal resolution, reducing noise for the GAT stage.

Architecture (per node, per subject):
    Input  : (B*N, 1, T)
    Three parallel Conv1d branches (k=3, k=5, k=7) → each: Conv → BN → ReLU → MaxPool(2)
    Concatenate along channel dim → (B*N, 3*base_ch, T//2)  [same-padding after pool]
    AdaptiveAvgPool1d(1) → squeeze → Linear → LayerNorm → Dropout → ReLU
    Output : (B*N, node_feat)  →  reshaped to (B, N, node_feat)

Also contains bold_signals_to_tensor() – identical helper to gnn_cnn_harmonised.
"""

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Helper: list[np.ndarray(T,N)] → padded tensor (B, N, T_max)
# ─────────────────────────────────────────────────────────────────────────────
def bold_signals_to_tensor(bold_signals: list, min_t: int = None) -> torch.Tensor:
    """
    Convert a list of (T_i, N_ROIs) arrays into a single (B, N, T_max)
    float32 tensor, zero-padding shorter scans.

    Assumes Z-scoring has already been applied by the caller (train.py).
    """
    valid = [b for b in bold_signals if b is not None]
    if not valid:
        raise RuntimeError("No valid BOLD signals found.")

    if min_t is not None:
        valid = [b for b in valid if b.shape[0] >= min_t]
        if not valid:
            raise RuntimeError(f"No subjects with T >= {min_t}.")

    T_max = max(b.shape[0] for b in valid)
    N     = valid[0].shape[1]

    arrays = []
    for b in valid:
        T_i = b.shape[0]
        if T_i < T_max:
            pad = np.zeros((T_max - T_i, N), dtype=np.float32)
            b   = np.concatenate([b, pad], axis=0)
        arrays.append(b)

    stacked = np.stack(arrays, axis=0)          # (B, T_max, N)
    stacked = stacked.transpose(0, 2, 1)        # (B, N, T_max)
    return torch.tensor(stacked, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale Inception CNN
# ─────────────────────────────────────────────────────────────────────────────
class MultiScaleInceptionCNN(nn.Module):
    """
    Multi-Scale Inception CNN temporal encoder.

    Three parallel 1-D convolutional branches operate at different temporal
    scales simultaneously, then their outputs are fused by concatenation:

        Branch A  k=3   → high-frequency transients
        Branch B  k=5   → mid-frequency HRF dynamics  (NYU TR=2000ms)
        Branch C  k=7   → slower haemodynamic fluctuations

    Parameters
    ----------
    n_regions : int
        Number of brain ROIs (N).  Used only for documentation / shape checks.
    node_feat : int
        Output feature dimension F per node (default 64).
    base_channels : int
        Output channels per branch before concatenation (default 32).
        Concatenated output has 3 * base_channels channels.
    """

    def __init__(
        self,
        n_regions: int = 116,
        node_feat: int = 64,
        base_channels: int = 32,
    ):
        super().__init__()
        self.n_regions    = n_regions
        self.node_feat     = node_feat
        self.base_channels = base_channels

        def _branch(kernel: int) -> nn.Sequential:
            """Single Inception branch: Conv1d → BN → ReLU → MaxPool(2), same-padding."""
            pad = kernel // 2          # same-length convolution
            return nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=base_channels,
                    kernel_size=kernel,
                    padding=pad,
                    bias=False,
                ),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
                # Temporal Max Pooling: captures "bursts", halves temporal dim
                nn.MaxPool1d(kernel_size=2, stride=2),
            )

        # Parallel branches — NYU-tuned kernels (3, 5, 7)
        self.branch3 = _branch(3)
        self.branch5 = _branch(5)
        self.branch7 = _branch(7)

        fused_channels = 3 * base_channels           # after cat along dim=1

        # Adaptive pooling → fixed-size independent of T
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

        # Projection to node feature space
        self.proj = nn.Sequential(
            nn.Linear(fused_channels, node_feat),
            nn.LayerNorm(node_feat),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, N, T)

        Returns
        -------
        torch.Tensor  shape (B, N, node_feat)
        """
        B, N, T = x.shape

        # Fold batch and node dims → treat every ROI independently
        x = x.reshape(B * N, 1, T)                  # (B*N, 1, T)

        # Three parallel branches — NYU-tuned
        a = self.branch3(x)                          # (B*N, base_ch, T')
        b = self.branch5(x)                          # (B*N, base_ch, T')
        c = self.branch7(x)                          # (B*N, base_ch, T')

        # Fuse by concatenation along channel dim
        fused = torch.cat([a, b, c], dim=1)          # (B*N, 3*base_ch, T')

        # Adaptive average pooling → (B*N, 3*base_ch, 1)
        fused = self.pool(fused)
        fused = fused.squeeze(-1)                    # (B*N, 3*base_ch)

        # Project to node feature space
        out = self.proj(fused)                       # (B*N, node_feat)

        # Unflatten → (B, N, node_feat)
        out = out.reshape(B, N, self.node_feat)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, N, T     = 4, 28, 175   # 28 social-brain ROIs
    node_feat   = 64

    x     = torch.randn(B, N, T)
    model = MultiScaleInceptionCNN(n_regions=N, node_feat=node_feat)
    model.eval()

    with torch.no_grad():
        out = model(x)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {out.shape}   (expected: ({B}, {N}, {node_feat}))")
    assert out.shape == (B, N, node_feat), "Shape mismatch!"
    print("MultiScaleInceptionCNN smoke-test passed [OK]")
