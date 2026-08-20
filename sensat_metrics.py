"""
metrics.py — Evaluation Metrics for Generative 3D Point Cloud Completion.

Implements a clean PointCloudEvaluator class that computes 5 standard
quantitative metrics for evaluating a VAE-based 3D point cloud completion model
where k=10 completion variants are generated per partial input shape.

Metrics Implemented:
    - CD  : Chamfer Distance           (point-level geometry accuracy)
    - EMD : Earth Mover's Distance     (point-level transportation cost)
    - MMD : Minimum Matching Distance  (generation fidelity to ground truth)
    - TMD : Total Mutual Difference    (generation diversity across k variants)
    - JSD : Jensen-Shannon Divergence  (distributional similarity, 28^3 voxels)

References:
    - Wu et al. (2020) "Multimodal Shape Completion"  — MMD, TMD, JSD over 28^3 voxels
    - HyperPocket (losses/champfer_loss.py)            — CD via batch_pairwise_dist (torch.bmm)

Usage:
    evaluator = PointCloudEvaluator(device='cuda')
    results = evaluator.evaluate(
        generated = torch.rand(10, 1024, 3),  # k=10 generated completions [k, N, 3]
        reference = torch.rand(50, 1024, 3),  # ground-truth reference set  [M, N, 3]
    )
    # results = {"CD": ..., "EMD": ..., "MMD": ..., "TMD": ..., "JSD": ...}
"""

import torch
import torch.nn as nn
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
VOXEL_GRID_DIM = 28         # 28^3 discrete voxel grid for JSD (Wu et al., 2020)
K_VARIANTS     = 10         # Number of completion variants generated per partial input
EMD_EPSILON    = 0.01       # Sinkhorn regularisation strength
EMD_MAX_ITER   = 50         # Sinkhorn iteration cap


# ==========================================
# CHAMFER DISTANCE
# Copied directly from HyperPocket's native:
#   repos/Hyperpocket .../losses/champfer_loss.py
# Uses torch.bmm to compute pairwise squared distances via the identity:
#   ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x . y^T
# ==========================================
class _ChamferLoss(nn.Module):
    """Pure PyTorch Chamfer Distance — copied verbatim from HyperPocket's
    losses/champfer_loss.py and documented here for thesis clarity.

    For two point clouds X (shape [B, M, 3]) and Y (shape [B, N, 3]):
        CD(X, Y) = sum_{x in X} min_{y in Y} ||x - y||^2
                 + sum_{y in Y} min_{x in X} ||x - y||^2
    """

    def __init__(self):
        super(_ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        """Compute summed Chamfer Distance between two batches of point clouds.

        Args:
            preds (Tensor): Predicted point cloud,   shape [B, N, 3].
            gts   (Tensor): Ground-truth point cloud, shape [B, M, 3].

        Returns:
            Tensor: Scalar CD loss (summed over batch and points).
        """
        P = self.batch_pairwise_dist(gts, preds)
        # Term 1: for each point in gts, nearest in preds
        mins, _ = torch.min(P, 1)
        loss_1  = torch.sum(mins)
        # Term 2: for each point in preds, nearest in gts
        mins, _ = torch.min(P, 2)
        loss_2  = torch.sum(mins)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        """Compute pairwise squared L2 distance matrix via torch.bmm.

        Uses the algebraic identity:
            ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 * <x_i, y_j>

        Args:
            x (Tensor): shape [B, M, 3]
            y (Tensor): shape [B, N, 3]

        Returns:
            P (Tensor): Pairwise distance matrix, shape [B, M, N].
        """
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _           = y.size()

        # Self-dot products for squared norms
        xx = torch.bmm(x, x.transpose(2, 1))   # [B, M, M]
        yy = torch.bmm(y, y.transpose(2, 1))   # [B, N, N]
        zz = torch.bmm(x, y.transpose(2, 1))   # [B, M, N]  cross term

        # Extract only the diagonal (squared norms of each point)
        dtype      = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)

        # Expand norms to [B, M, N] for broadcasting
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)

        # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 <x_i, y_j>
        P = rx.transpose(2, 1) + ry - 2 * zz   # [B, M, N]
        return P


# ==========================================
# EARTH MOVER'S DISTANCE (SINKHORN APPROXIMATION)
# Batched Sinkhorn-Knopp optimal transport implemented entirely in PyTorch.
# Avoids C++/CUDA extensions and cluster compilation errors.
# Sinkhorn approximates EMD using entropy-regularized OT:
#   EMD_eps(X, Y) ≈ min_{T in Pi(mu, nu)} sum_{ij} T_ij * C_ij
#                   + eps * KL(T || mu ⊗ nu)
# ==========================================
def _sinkhorn_emd(x, y, epsilon=EMD_EPSILON, max_iter=EMD_MAX_ITER):
    """Compute Sinkhorn optimal transport distance between two point clouds.

    Approximates EMD using entropy-regularised optimal transport (Sinkhorn-Knopp
    algorithm). Runs fully in PyTorch — no C++/CUDA extensions required.

    Args:
        x       (Tensor): Source point cloud, shape [B, N, 3].
        y       (Tensor): Target point cloud, shape [B, M, 3].
        epsilon (float) : Sinkhorn regularisation strength. Smaller = closer to true EMD.
        max_iter(int)   : Number of Sinkhorn iterations.

    Returns:
        Tensor: Per-sample Sinkhorn EMD distances, shape [B].
    """
    B, N, _ = x.shape
    _, M, _ = y.shape

    # Step 1: Compute pairwise cost matrix C using squared L2 distance
    # C[b, i, j] = ||x[b,i] - y[b,j]||^2
    x_expand = x.unsqueeze(2).expand(B, N, M, 3)   # [B, N, M, 3]
    y_expand = y.unsqueeze(1).expand(B, N, M, 3)   # [B, N, M, 3]
    C = torch.sum((x_expand - y_expand) ** 2, dim=-1)   # [B, N, M]

    # Step 2: Uniform marginals (equal weight to each point)
    mu = torch.ones(B, N, device=x.device, dtype=x.dtype) / N   # [B, N]
    nu = torch.ones(B, M, device=x.device, dtype=x.dtype) / M   # [B, M]

    # Step 3: Gibbs kernel K = exp(-C / epsilon)
    log_K = -C / epsilon   # [B, N, M]

    # Step 4: Sinkhorn iterations (log-domain for numerical stability)
    log_u = torch.zeros(B, N, device=x.device, dtype=x.dtype)   # [B, N]
    log_v = torch.zeros(B, M, device=x.device, dtype=x.dtype)   # [B, M]

    log_mu = torch.log(mu + 1e-8)
    log_nu = torch.log(nu + 1e-8)

    for _ in range(max_iter):
        # u-update: log_u = log_mu - logsumexp(log_K + log_v, dim=2)
        log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
        # v-update: log_v = log_nu - logsumexp(log_K + log_u, dim=1)
        log_v = log_nu - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)

    # Step 5: Compute optimal transport plan T and extract scalar EMD
    # T[b, i, j] = exp(log_u[b,i] + log_K[b,i,j] + log_v[b,j])
    log_T = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)   # [B, N, M]
    T     = torch.exp(log_T)                                    # [B, N, M]

    # EMD = sum_{ij} T_ij * C_ij  (scalar per sample in batch)
    emd_per_sample = (T * C).sum(dim=[1, 2])   # [B]
    return emd_per_sample


# ==========================================
# VOXELIZATION HELPER FOR JSD
# Converts a point cloud in [-1, 1]^3 into a 28^3 occupancy histogram.
# As per Wu et al. (2020): bin coordinates into VOXEL_GRID_DIM^3 discrete cells.
# ==========================================
def _voxelize(points_batch, grid_dim=VOXEL_GRID_DIM):
    """Convert a batch of unit-sphere point clouds into flat 28^3 voxel histograms.

    Our SensatUrban blocks are pre-normalized to [-1, 1]^3, so we directly
    map coordinates to voxel indices: idx = floor((coord + 1) / 2 * grid_dim)
    and clamp to [0, grid_dim-1] for numerical safety at the boundary.

    Args:
        points_batch (Tensor): Point clouds, shape [B, N, 3], coords in [-1, 1].
        grid_dim     (int)   : Voxel grid side length (28 per Wu et al., 2020).

    Returns:
        np.ndarray: Normalised voxel histograms, shape [B, grid_dim^3].
    """
    B, N, _ = points_batch.shape
    pts     = points_batch.detach().cpu().numpy()   # [B, N, 3]

    # Map from [-1, 1] → [0, grid_dim-1] integer voxel indices
    indices = np.floor((pts + 1.0) / 2.0 * grid_dim).astype(np.int32)
    indices = np.clip(indices, 0, grid_dim - 1)     # safety clamp at boundary

    total_voxels = grid_dim ** 3
    histograms   = np.zeros((B, total_voxels), dtype=np.float64)

    for b in range(B):
        # Pack 3D index (ix, iy, iz) into a single flat index
        flat_idx = (indices[b, :, 0] * grid_dim * grid_dim +
                    indices[b, :, 1] * grid_dim +
                    indices[b, :, 2])
        # Count occupancy per voxel and normalise to a probability distribution
        np.add.at(histograms[b], flat_idx, 1.0)
        total = histograms[b].sum()
        if total > 0:
            histograms[b] /= total

    return histograms   # [B, grid_dim^3]


# ==========================================
# MAIN EVALUATOR CLASS
# ==========================================
class PointCloudEvaluator:
    """Computes CD, EMD, MMD, TMD, and JSD for 3D point cloud completion models.

    Designed for models (e.g. HyperPocket) that generate k=10 diverse
    completion variants per partial input shape.

    All computation runs on GPU (or CPU if CUDA unavailable). Inputs must be
    PyTorch FloatTensors with coordinates normalised to a unit sphere [-1, 1]^3.

    Args:
        device  (str) : 'cuda' or 'cpu'. Defaults to auto-detect.
        k       (int) : Number of generated completion variants per input (default: 10).
        verbose (bool): If True, prints metric values as they are computed.

    Example:
        evaluator = PointCloudEvaluator(device='cuda')
        results = evaluator.evaluate(
            generated = torch.rand(10, 1024, 3).cuda(),  # [k, N, 3]
            reference = torch.rand(50, 1024, 3).cuda(),  # [M, N, 3]
        )
        print(results)
        # {"CD": 0.031, "EMD": 0.012, "MMD": 0.027, "TMD": 0.188, "JSD": 0.043}
    """

    def __init__(self, device=None, k=K_VARIANTS, verbose=True):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device  = torch.device(device)
        self.k       = k
        self.verbose = verbose

        # Instantiate the HyperPocket-native Chamfer Distance module
        self._cd_fn = _ChamferLoss().to(self.device)

    # ------------------------------------------------------------------
    # INTERNAL: Pairwise CD between two single point clouds (unbatched)
    # ------------------------------------------------------------------
    def _cd_pair(self, x, y):
        """Compute CD between two single point clouds x and y.

        Args:
            x (Tensor): shape [N, 3]
            y (Tensor): shape [M, 3]

        Returns:
            float: Scalar Chamfer Distance value.
        """
        # Add batch dimension for _ChamferLoss which expects [B, N, 3]
        x_b = x.unsqueeze(0)   # [1, N, 3]
        y_b = y.unsqueeze(0)   # [1, M, 3]
        return self._cd_fn(x_b, y_b).item()

    # ------------------------------------------------------------------
    # 1. CHAMFER DISTANCE (CD)
    # ------------------------------------------------------------------
    def compute_cd(self, generated, reference):
        """Compute mean Chamfer Distance between generated and reference sets.

        Averages CD over all (generated, reference) pair combinations.
        Useful as a global shape accuracy scalar.

        Args:
            generated (Tensor): Generated completions, shape [k, N, 3].
            reference (Tensor): Ground-truth references, shape [M, N, 3].

        Returns:
            float: Mean CD across all pairs.
        """
        total, count = 0.0, 0
        for g in generated:
            for r in reference:
                total += self._cd_pair(g, r)
                count += 1
        cd = total / count if count > 0 else 0.0
        if self.verbose:
            print(f"  CD  = {cd:.6f}")
        return cd

    # ------------------------------------------------------------------
    # 2. EARTH MOVER'S DISTANCE (EMD)
    # ------------------------------------------------------------------
    def compute_emd(self, generated, reference):
        """Compute mean Sinkhorn EMD between generated and reference sets.

        Uses the entropy-regularised Sinkhorn-Knopp algorithm implemented
        purely in PyTorch — no C++/CUDA extension required.

        Args:
            generated (Tensor): Generated completions, shape [k, N, 3].
            reference (Tensor): Ground-truth references, shape [M, N, 3].

        Returns:
            float: Mean Sinkhorn EMD across all (generated, reference) pairs.
        """
        total, count = 0.0, 0
        for g in generated:
            for r in reference:
                # Expand to batched [1, N, 3] and compute
                emd_val = _sinkhorn_emd(
                    g.unsqueeze(0), r.unsqueeze(0),
                    epsilon=EMD_EPSILON, max_iter=EMD_MAX_ITER
                )
                total += emd_val.item()
                count += 1
        emd = total / count if count > 0 else 0.0
        if self.verbose:
            print(f"  EMD = {emd:.6f}")
        return emd

    # ------------------------------------------------------------------
    # 3. MINIMUM MATCHING DISTANCE (MMD)
    # ------------------------------------------------------------------
    def compute_mmd(self, generated, reference):
        """Compute Minimum Matching Distance (MMD) measuring generation fidelity.

        For each reference shape Y in Sr, finds the closest generated shape
        X in Sg by CD, and averages these minimum distances.

        Formula (Wu et al., 2020):
            MMD(Sg, Sr) = (1 / |Sr|) * sum_{Y in Sr} min_{X in Sg} CD(X, Y)

        Args:
            generated (Tensor): Generated completions, shape [k, N, 3].
            reference (Tensor): Ground-truth references, shape [M, N, 3].

        Returns:
            float: MMD value. Lower = generated shapes are closer to real shapes.
        """
        total = 0.0
        for r in reference:
            # For each reference shape, find the minimum CD among all k generated shapes
            min_cd = min(self._cd_pair(g, r) for g in generated)
            total += min_cd

        mmd = total / len(reference)
        if self.verbose:
            print(f"  MMD = {mmd:.6f}")
        return mmd

    # ------------------------------------------------------------------
    # 4. TOTAL MUTUAL DIFFERENCE (TMD)
    # ------------------------------------------------------------------
    def compute_tmd(self, generated):
        """Compute Total Mutual Difference (TMD) measuring diversity across k variants.

        For each generated variant X_i, computes its mean CD to all other k-1
        variants. Sums these mean pairwise distances across all k variants.

        Formula (Wu et al., 2020):
            TMD = sum_{i=1}^{k} (1 / (k-1)) * sum_{j != i} CD(X_i, X_j)

        Args:
            generated (Tensor): k generated completions, shape [k, N, 3].

        Returns:
            float: TMD value. Higher = more diverse generated completions.
        """
        k   = len(generated)
        tmd = 0.0

        if k < 2:
            if self.verbose:
                print(f"  TMD = 0.000000  (k < 2, diversity undefined)")
            return 0.0

        for i in range(k):
            row_sum = sum(
                self._cd_pair(generated[i], generated[j])
                for j in range(k) if j != i
            )
            tmd += row_sum / (k - 1)

        if self.verbose:
            print(f"  TMD = {tmd:.6f}")
        return tmd

    # ------------------------------------------------------------------
    # 5. JENSEN-SHANNON DIVERGENCE (JSD) over 28^3 voxel grid
    # ------------------------------------------------------------------
    def compute_jsd(self, generated, reference):
        """Compute Jensen-Shannon Divergence (JSD) over a 28^3 voxel grid.

        As per Wu et al. (2020), both sets are voxelised into a 28^3 discrete
        grid to obtain marginal probability distributions P and Q over occupied
        voxels. JSD is then computed as:

            M   = 0.5 * (P + Q)
            JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
                = 0.5 * sum(P * log(P / M)) + 0.5 * sum(Q * log(Q / M))

        Tiny epsilon (1e-8) prevents log(0) errors.

        Our SensatUrban blocks are pre-normalised to [-1, 1]^3, so voxel
        indices are directly derived from coordinates without re-scaling.

        Args:
            generated (Tensor): Generated completions, shape [k, N, 3].
            reference (Tensor): Ground-truth references, shape [M, N, 3].

        Returns:
            float: JSD in [0, log(2)]. Lower = distributions are more similar.
        """
        eps = 1e-8

        # Voxelise both sets → normalised occupancy histograms
        hist_gen = _voxelize(generated, grid_dim=VOXEL_GRID_DIM)   # [k, 28^3]
        hist_ref = _voxelize(reference, grid_dim=VOXEL_GRID_DIM)   # [M, 28^3]

        # Aggregate into single global distributions by averaging across samples
        P = hist_gen.mean(axis=0)   # [28^3]
        Q = hist_ref.mean(axis=0)   # [28^3]

        # Mixture distribution M = 0.5 * (P + Q)
        M = 0.5 * (P + Q)

        # KL(P || M) = sum(P * log(P / M))
        kl_pm = np.sum(P * np.log((P + eps) / (M + eps)))
        # KL(Q || M) = sum(Q * log(Q / M))
        kl_qm = np.sum(Q * np.log((Q + eps) / (M + eps)))

        # JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        jsd = 0.5 * kl_pm + 0.5 * kl_qm

        if self.verbose:
            print(f"  JSD = {jsd:.6f}")
        return float(jsd)

    # ------------------------------------------------------------------
    # UNIFIED EVALUATE METHOD
    # ------------------------------------------------------------------
    def evaluate(self, generated, reference):
        """Compute all 5 metrics in one call and return a result dictionary.

        Args:
            generated (Tensor): k generated completion variants, shape [k, N, 3].
                                 Coords must be normalised to unit sphere [-1, 1]^3.
            reference (Tensor): Ground-truth reference set, shape [M, N, 3].
                                 Coords must be normalised to unit sphere [-1, 1]^3.

        Returns:
            dict: {
                "CD"  : float — Chamfer Distance (mean over all pairs),
                "EMD" : float — Earth Mover's Distance (mean Sinkhorn, over all pairs),
                "MMD" : float — Minimum Matching Distance (fidelity),
                "TMD" : float — Total Mutual Difference (diversity),
                "JSD" : float — Jensen-Shannon Divergence over 28^3 voxels,
            }
        """
        generated = generated.to(self.device)
        reference = reference.to(self.device)

        if self.verbose:
            print(f"\n[PointCloudEvaluator] Computing metrics ...")
            print(f"  Generated : {generated.shape}   ({self.k} variants)")
            print(f"  Reference : {reference.shape}")
            print(f"  Voxel grid: {VOXEL_GRID_DIM}^3 = {VOXEL_GRID_DIM**3} cells\n")

        results = {
            "CD" : self.compute_cd(generated, reference),
            "EMD": self.compute_emd(generated, reference),
            "MMD": self.compute_mmd(generated, reference),
            "TMD": self.compute_tmd(generated),
            "JSD": self.compute_jsd(generated, reference),
        }

        if self.verbose:
            print(f"\n  {'='*40}")
            print(f"  Evaluation Summary:")
            for k_name, v in results.items():
                print(f"    {k_name:<5} = {v:.6f}")
            print(f"  {'='*40}\n")

        return results


# ==========================================
# QUICK STANDALONE TEST
# ==========================================
if __name__ == "__main__":
    print("\n--- PointCloudEvaluator Sanity Test (using random tensors) ---\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}\n")

    # Simulate k=10 generated completions and M=50 reference shapes
    # Shape [B, N, 3] with coordinates in [-1, 1] (unit sphere — like SensatUrban output)
    torch.manual_seed(42)
    generated = torch.rand(10, 1024, 3) * 2 - 1   # [10, 1024, 3], range [-1, 1]
    reference = torch.rand(50, 1024, 3) * 2 - 1   # [50, 1024, 3], range [-1, 1]

    evaluator = PointCloudEvaluator(device=device, k=10, verbose=True)
    results   = evaluator.evaluate(generated, reference)

    print("Raw results dictionary:")
    print(results)
    print("\n--- Sanity test complete ---")
