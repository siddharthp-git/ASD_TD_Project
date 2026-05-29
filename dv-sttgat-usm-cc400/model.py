"""
model.py – DV-STTGAT  (NYU-only, SOTA v3)
==========================================
Dual-View Spatio-Temporal Graph Attention Transformer — single-site variant
with 5 SOTA upgrades applied:

  1. Social-Brain ROI pruning  → handled upstream in data_loader.py
  2. LearnableGraph (SLAMP)    → cosine-similarity adjacency with top-k sparsity
  3. Slim Spatial Branch       → gat_hidden=16, three-view learnable gating
  4. Temporal Max Pooling      → applied in cnn.py before this module
  5. Attention Global Pooling  → replaces ViT CLS-token + TransformerEncoder
  encode() / classify() split  → enables Manifold Mixup in train.py

Architecture
------------
1. Temporal Branch   : MultiScaleInceptionCNN  →  node features (B*N, F)
                       (MaxPool already applied inside CNN)
2. Spatial Branch    : Triple-Branch GATv2
   - Path A: GATv2Conv on G_pear (static Pearson edges)
   - Path B: GATv2Conv on G_prec (static Precision edges)
   - Path C: GATv2Conv on LearnableGraph edges (cosine-similarity, top-k)
   - Residual: H_out = GAT(H_in) + Linear(H_in)
   - Fuse: learnable softmax gating over the 3 paths
3. Global Readout    : Attention-based Global Pooling (per-node score → weighted sum)
4. Classifier        : Linear projection on pooled graph embedding

DANN / GRL removed — single NYU site needs no domain adaptation.
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from cnn import MultiScaleInceptionCNN


# ─────────────────────────────────────────────────────────────────────────────
# Learnable Graph Module (SLAMP — cosine-similarity adjacency)
# ─────────────────────────────────────────────────────────────────────────────
class LearnableGraph(nn.Module):
    """
    Constructs a sparse graph from node features using cosine similarity.

    Rationale
    ---------
    BOLD signals are amplitude-sensitive; Euclidean distance would conflate
    signal intensity with functional coupling.  Cosine similarity is purely
    direction-based and captures synchronisation rhythm — the actual ASD
    biomarker — while ignoring amplitude differences.

    Algorithm (per graph in the batch)
    ------------------------------------
    1. Project CNN features: z = L2_norm(Linear(x))   →  (N, 32)
    2. Cosine sim matrix:    S = z @ z.T               →  (N, N)
    3. Soft adj:             A = exp(S / τ)            (τ learnable, init=0.5)
    4. Top-k sparsify:       keep k highest-sim edges per node
    5. Return (edge_index, edge_weight) compatible with GATv2Conv

    Parameters
    ----------
    in_features : int   – CNN output feature dim (F)
    proj_dim    : int   – projection dim (default 32)
    k           : int   – top-k edges per node (default 10)
    """

    def __init__(self, in_features: int, proj_dim: int = 32, k: int = 10):
        super().__init__()
        self.k    = k
        self.proj = nn.Linear(in_features, proj_dim, bias=False)
        # Learnable temperature — controls sharpness of the soft adjacency
        self.temperature = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        """
        Parameters
        ----------
        x     : (B*N, F)  CNN node features (already LayerNorm'd)
        batch : (B*N,)    PyG batch vector

        Returns
        -------
        edge_index : LongTensor  (2, E_learn)
        edge_attr  : FloatTensor (E_learn,)
        """
        # Project to lower-dim space and L2-normalise → cosine similarity
        z = F.normalize(self.proj(x), p=2, dim=-1)   # (B*N, proj_dim)

        # Convert flat node features to dense (B, N, proj_dim) for batch-wise ops
        z_dense, mask = to_dense_batch(z, batch)      # (B, N_max, proj_dim)
        B, N_max, D   = z_dense.shape

        tau = torch.clamp(self.temperature, min=1e-2)  # prevent collapse

        all_ei, all_ew = [], []

        for b in range(B):
            # Valid nodes only (mask out padding if any)
            m   = mask[b]                              # (N_max,)
            z_b = z_dense[b][m]                       # (N, proj_dim)
            N   = z_b.size(0)

            if N <= 1:
                # Edge case: single node graph → no edges
                ei = torch.zeros(2, 0, dtype=torch.long,  device=x.device)
                ew = torch.zeros(0,    dtype=torch.float32, device=x.device)
                all_ei.append(ei)
                all_ew.append(ew)
                continue

            # Cosine similarity: (N, N)
            sim = z_b @ z_b.T                         # already L2-normed → cosine sim

            # Zero out self-loops BEFORE exp() — avoids inplace modification of
            # the exp output tensor, which would corrupt autograd backward pass.
            off_diag = ~torch.eye(N, dtype=torch.bool, device=x.device)
            sim = sim * off_diag                      # non-inplace: self-sim → 0

            # Soft adjacency via exponential scaling
            adj = torch.exp(sim / tau)                # (N, N)  — never touched inplace

            # Top-k sparsification: for each node keep at most k neighbours
            k_eff = min(self.k, N - 1)
            topk_vals, topk_cols = adj.topk(k_eff, dim=-1)  # (N, k)

            rows = torch.arange(N, device=x.device).unsqueeze(-1).expand_as(topk_cols)
            rows = rows.reshape(-1)
            cols = topk_cols.reshape(-1)
            ews  = topk_vals.reshape(-1)

            # Offset node indices by the start of this graph in the flat batch
            offset = (batch == b).nonzero(as_tuple=True)[0][0]
            ei = torch.stack([rows + offset, cols + offset], dim=0)
            all_ei.append(ei)
            all_ew.append(ews)

        edge_index = torch.cat(all_ei, dim=1)         # (2, E_learn)
        edge_attr  = torch.cat(all_ew, dim=0)         # (E_learn,)
        return edge_index, edge_attr


# ─────────────────────────────────────────────────────────────────────────────
# Dual-View Spatio-Temporal Graph Attention Transformer  (NYU-only, SOTA v3)
# ─────────────────────────────────────────────────────────────────────────────
class DVSTTGATModel(nn.Module):
    """
    Parameters
    ----------
    n_regions         : Number of brain ROIs (N).  Default = 28 (social-brain subset).
    temporal_out_feat : CNN output feature dim per node (F, default 64).
    gat_hidden        : Feature dim inside each GAT head (H, default 16 — slim).
    gat_heads         : Number of GATv2 attention heads (default 4).
    num_sites         : Kept for API compatibility; DANN head is removed.
    num_classes       : 1 for binary classification.
    dropout           : Shared dropout rate.
    learn_k           : Top-k for LearnableGraph cosine-sim edges (default 10).
    """

    def __init__(
        self,
        n_regions: int = 28,           # 28 social-brain ROIs after pruning
        temporal_out_feat: int = 64,
        gat_hidden: int = 16,          # Slim spatial branch: 64→16
        gat_heads: int = 4,
        num_sites: int = 1,            # kept for API compat, DANN removed
        num_classes: int = 1,
        dropout: float = 0.5,
        learn_k: int = 10,
    ):
        super().__init__()
        self.n_regions         = n_regions
        self.temporal_out_feat = temporal_out_feat
        self.gat_hidden        = gat_hidden
        H = gat_hidden

        # ── 1. Temporal CNN ───────────────────────────────────────────────────
        self.temporal_cnn = MultiScaleInceptionCNN(
            n_regions=n_regions,
            node_feat=temporal_out_feat,
        )
        # Normalise per-node CNN features → stabilises GAT input across folds
        self.cnn_norm = nn.LayerNorm(temporal_out_feat)

        # ── 2. LearnableGraph (cosine-similarity SLAMP) ───────────────────────
        self.learnable_graph = LearnableGraph(
            in_features=temporal_out_feat, proj_dim=32, k=learn_k
        )

        # ── 3a. Triple-Branch GATv2 (first layer) ────────────────────────────
        _gat1_kwargs = dict(
            in_channels=temporal_out_feat,
            out_channels=H,
            heads=gat_heads,
            concat=True,
            dropout=0.3,
            edge_dim=1,
        )
        self.gat_pear1  = GATv2Conv(**_gat1_kwargs)   # Path A: Pearson
        self.gat_prec1  = GATv2Conv(**_gat1_kwargs)   # Path B: Precision
        self.gat_learn1 = GATv2Conv(**_gat1_kwargs)   # Path C: Learnable

        # ── 3b. Triple-Branch GATv2 (second layer) ───────────────────────────
        _gat2_kwargs = dict(
            in_channels=H * gat_heads,
            out_channels=H,
            heads=gat_heads,
            concat=False,  # average heads → output (B*N, H)
            dropout=0.3,
            edge_dim=1,
        )
        self.gat_pear2  = GATv2Conv(**_gat2_kwargs)
        self.gat_prec2  = GATv2Conv(**_gat2_kwargs)
        self.gat_learn2 = GATv2Conv(**_gat2_kwargs)

        # ── 3c. Residual projection (F → H, shared across all paths) ─────────
        self.residual_proj = nn.Linear(temporal_out_feat, H)

        # ── 3d. Learnable view-gating (3 views: Pearson, Precision, Learned) ─
        # Softmax over these 3 weights determines per-dataset trust per view.
        self.view_weights = nn.Parameter(torch.ones(3))

        # Normalise fused GAT output → prevents fold-specific gradient drift
        self.gat_norm = nn.LayerNorm(H)

        # ── 4. Attention-based Global Pooling ─────────────────────────────────
        # Replaces ViT CLS-token + TransformerEncoder.
        # Scores each ROI's contribution to the final diagnosis.
        self.attn_pool = nn.Linear(H, 1)

        # ── 5. Classification head ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(H, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes),
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _run_triple_gat(self, x: torch.Tensor, batch_data, batch: torch.Tensor):
        """
        Run all three GATv2 paths and fuse with learnable softmax gating.

        x          : (B*N, F)  CNN node features
        batch_data : PyG Batch
        batch      : (B*N,)  batch vector

        Returns (B*N, H) fused node embeddings.
        """
        ew_pear = torch.abs(batch_data.edge_attr_pear).unsqueeze(-1)  # (E_pear, 1)
        ew_prec = torch.abs(batch_data.edge_attr_prec).unsqueeze(-1)  # (E_prec, 1)

        residual = self.residual_proj(x)   # (B*N, H)

        # ── Path A: Pearson graph ─────────────────────────────────────────────
        a = self.gat_pear1(x, batch_data.edge_index_pear, edge_attr=ew_pear)
        a = F.elu(a)
        a = F.dropout(a, p=0.3, training=self.training)
        a = self.gat_pear2(a, batch_data.edge_index_pear, edge_attr=ew_pear)
        a = F.elu(a) + residual

        # ── Path B: Precision / partial-correlation graph ─────────────────────
        b = self.gat_prec1(x, batch_data.edge_index_prec, edge_attr=ew_prec)
        b = F.elu(b)
        b = F.dropout(b, p=0.3, training=self.training)
        b = self.gat_prec2(b, batch_data.edge_index_prec, edge_attr=ew_prec)
        b = F.elu(b) + residual

        # ── Path C: Learnable cosine-similarity graph ─────────────────────────
        learn_ei, learn_ew = self.learnable_graph(x, batch)
        learn_ew = learn_ew.unsqueeze(-1)                   # (E_learn, 1)
        c = self.gat_learn1(x, learn_ei, edge_attr=learn_ew)
        c = F.elu(c)
        c = F.dropout(c, p=0.3, training=self.training)
        c = self.gat_learn2(c, learn_ei, edge_attr=learn_ew)
        c = F.elu(c) + residual

        # ── Learnable softmax gating over the 3 views ─────────────────────────
        gates = F.softmax(self.view_weights, dim=0)          # (3,)
        fused = gates[0] * a + gates[1] * b + gates[2] * c  # (B*N, H)
        return fused

    # ─────────────────────────────────────────────────────────────────────────
    def encode(self, batch_data) -> torch.Tensor:
        """
        Full encoder: CNN → Triple GATv2 → Attention Global Pooling.

        Returns
        -------
        graph_emb : (B, H)  — one embedding per subject, ready for classifier.
        Used by train.py for Manifold Mixup (mixup applied on this embedding).
        """
        x     = batch_data.x       # (B*N, T_win)
        batch = batch_data.batch   # (B*N,)

        B = int(batch.max().item()) + 1
        N = self.n_regions
        T = x.size(1)

        # ── Step 1: Temporal CNN ──────────────────────────────────────────────
        x = x.view(B, N, T)                              # (B, N, T)
        x = self.temporal_cnn(x)                         # (B, N, F)
        x = x.view(B * N, self.temporal_out_feat)        # (B*N, F)
        x = self.cnn_norm(x)                             # LayerNorm

        # ── Step 2: Triple-Branch GATv2 with learnable gating ─────────────────
        x = self._run_triple_gat(x, batch_data, batch)   # (B*N, H)
        x = self.gat_norm(x)                             # LayerNorm

        # ── Step 3: Attention Global Pooling ──────────────────────────────────
        # Reshape to (B, N, H) for per-graph attention
        H = self.gat_hidden
        x = x.view(B, N, H)                              # (B, N, H)

        # Per-node attention score → softmax over nodes
        scores = self.attn_pool(x)                       # (B, N, 1)
        scores = F.softmax(scores, dim=1)                # (B, N, 1)

        # Weighted sum across ROIs → graph-level embedding
        graph_emb = (scores * x).sum(dim=1)              # (B, H)
        return graph_emb

    # ─────────────────────────────────────────────────────────────────────────
    def classify(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        Classification head only.  Separated for Manifold Mixup support.

        Parameters
        ----------
        graph_emb : (B, H)

        Returns
        -------
        logits : (B, 1)
        """
        return self.classifier(graph_emb)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, batch_data) -> torch.Tensor:
        """
        End-to-end forward pass.  Backward-compatible entry point.

        Returns
        -------
        logits : (B, 1)
        """
        return self.classify(self.encode(batch_data))


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, N, T = 4, 28, 175    # 28 social-brain ROIs
    H = 16

    def _mock_edges(N, n_edges):
        src = torch.randint(0, N, (n_edges,))
        dst = torch.randint(0, N, (n_edges,))
        return torch.stack([src, dst]), torch.rand(n_edges) * 2 - 1

    data_list = []
    for i in range(B):
        # NOTE: Do NOT pre-offset edges here — Batch.from_data_list does it.
        ei_p, ew_p = _mock_edges(N, 200)
        ei_r, ew_r = _mock_edges(N, 120)
        d = Data(
            x               = torch.randn(N, T),
            edge_index_pear = ei_p,
            edge_attr_pear  = ew_p,
            edge_index_prec = ei_r,
            edge_attr_prec  = ew_r,
            y               = torch.tensor([float(i % 2)]),
        )
        data_list.append(d)

    batch = Batch.from_data_list(data_list)

    print("=" * 60)
    print("DV-STTGAT SOTA v3 (NYU-only) forward pass smoke-test …")
    model = DVSTTGATModel(
        n_regions=N,
        temporal_out_feat=64,
        gat_hidden=H,
        gat_heads=4,
    )
    model.eval()

    with torch.no_grad():
        # Test full forward
        logits    = model(batch)
        # Test encode/classify split (for Manifold Mixup)
        graph_emb = model.encode(batch)
        logits2   = model.classify(graph_emb)
        # Test view gates
        gates = torch.softmax(model.view_weights, dim=0)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  logits      : {logits.shape}      (expect [{B}, 1])")
    print(f"  graph_emb   : {graph_emb.shape}   (expect [{B}, {H}])")
    print(f"  view gates  : {gates.detach().numpy().round(3)}  (sum to 1)")
    print(f"  Parameters  : {n_params:,}")
    assert logits.shape == (B, 1),       "logits shape mismatch"
    assert graph_emb.shape == (B, H),   "graph_emb shape mismatch"
    assert abs(gates.sum().item() - 1.0) < 1e-5, "gates don't sum to 1"
    print("Forward pass [OK] -- DV-STTGAT SOTA v3 NYU pipeline complete.")
    print("=" * 60)
