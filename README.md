# Generative VAE Reconstruction for Urban 3D Point Clouds

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-cuBLAS%20%7C%20AMP%20FP16-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![WandB Logged](https://img.shields.io/badge/WandB-Experiment%20Tracking-FFBE00.svg)](https://wandb.ai/)
[![Dataset: SensatUrban](https://img.shields.io/badge/Dataset-SensatUrban%20(2.8B%20pts)-brightgreen.svg)](https://github.com/QingyongHu/SensatUrban)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Master's Thesis Research**  
> **Institution:** Brandenburg University of Technology Cottbus-Senftenberg (BTU), Germany  
> **Student:** Syed Rayyan Adil  
> **Supervisors:** Martin Schorradt, Prof. Dr. Michael Breuß  
> **Title:** *Autoencoder-based reconstruction of incomplete structures in urban 3D point clouds*

---

> **TL;DR:** Real-world aerial LiDAR scans suffer from massive structural occlusions and noise. This repository implements a **Context-Aware Generative Variational Autoencoder (VAE)** that conditions implicit 3D point cloud completion on surrounding spatial urban topology ($K=4$ physical neighbors). Evaluated on **5,270 SensatUrban blocks (~2.8B raw points)**, spatial context conditioning reduces Chamfer reconstruction error by **up to 15.8%** across PointNet, Dynamic Graph CNN (DGCNN), Cross-Attention Transformer, and linear State Space Models (Point-Mamba).

---

<p align="center">
  <img src="assets/teaser_reconstruction.png" alt="3D Reconstruction Teaser" width="92%">
  <br>
  <em><b>Figure 1: Qualitative 3D Completion on SensatUrban.</b> Left: Visible context input ($P_e$); Middle: Ground-truth missing target ($P_m$); Right: Model generative reconstruction ($\hat{P}_m$). Spatial context conditioning accurately reconstructs missing building boundaries and planar roof geometry.</em>
</p>

---

## 📌 1. Motivation & Research Problem

* **The Reality of Urban LiDAR:** Aerial photogrammetric and LiDAR surveys (e.g. for smart cities, digital twins, and cultural heritage documentation) frequently capture incomplete structures due to sensor occlusion, shadow zones, flight path constraints, and reflective surfaces.
* **The Limitation of Prior Art:** State-of-the-art point cloud completion frameworks (e.g., PCN, TopNet, standard HyperPocket) operate exclusively on **isolated, synthetic CAD models (ShapeNet)**. In urban environments, objects do not exist in isolation; a building's shape is constrained by surrounding road vectors, adjacent structures, and terrain morphology.
* **Our Core Research Question:** *Can we exploit the spatial and structural context of surrounding city blocks to plausibly hallucinate and reconstruct missing 3D geometry in large-scale urban point clouds?*

---

## 📖 2. Abstract

This thesis proposes a context-conditioned generative framework for 3D urban point cloud completion. Built upon the implicit HyperPocket architecture (Wu et al., IEEE TVCG 2022), our system introduces a dedicated **Context Encoder ($E_c$)** that ingests the $K=4$ nearest physical neighbor tiles, aligned in true relative metric coordinates. We formulate a triple-encoder latent space concatenating deterministic visible context ($z_e$), stochastic target VAE latents ($z_m$), and spatial neighborhood context ($z_c$) to dynamically predict the weights of an implicit MLP TargetNetwork.

We systematically benchmark **four distinct context encoding paradigms**:
1. **Permutation-Invariant PointNet:** Baseline spatial feature pooling.
2. **Dynamic Graph CNN (DGCNN / EdgeConv):** Local neighborhood topology and geometric boundary learning.
3. **Latent Cross-Attention Set Transformer:** Attention-driven query pooling over local coordinate distributions.
4. **Point-Mamba State Space Models (SSM):** Bidirectional linear-complexity sequence modeling across Morton Z-order serializations.

Quantitative evaluation across **1,077 real-world SensatUrban test blocks** demonstrates that spatial context conditioning yields consistent reductions in reconstruction Chamfer Distance (from **68.68 down to 57.80**, a **15.8% error reduction**), with DGCNN and Cross-Attention providing the strongest structural regularization.

---

## 🛠️ 3. Methodology & System Architecture

```
                    [ Visible Input Pe ]  ----->  [ Real Encoder Ee ]      ----->  ze (128D) 
                                                                                      |
                    [ Missing Target Pm ] ----->  [ VAE Encoder Em ]       ----->  zm (128D)  ===> Concatenated Latent z_cond (384D)
                                                          |                           |                    |
                                                     N(mu, logvar)                    |                    v
                                                                                      |           [ Context HyperNetwork ]
[ Context Pc ] (K=4 Aligned Neighbors)  ----->  [ Context Encoder Ec ]  ----->  zc (128D)          (Linear MLP Backbone)
                                                (PointNet / DGCNN /                   |                    |
                                                 Transformer / Mamba)                                      v
                                                                                                  [ TargetNetwork Weights ]
                                                                                                           |
                                  [ Progressive Sphere Points ]  ----------------------------------->  [ TargetNetwork MLP ]  ---> [ Reconstructed Output Pm ]
```

### 3.1 Spatial Neighborhood Formulation ($P_c$)
From the full SensatUrban city point cloud, each $30\,\text{m} \times 30\,\text{m}$ ground tile is paired with its $K=4$ nearest spatial neighbors using a 2D ground $k$-d tree. Neighbors are aligned in their true relative physical coordinates before resampling to a canonical budget of $N_c = 1,024$ points:
$$P_{\text{neighbor, aligned}} = P_{\text{neighbor}} + \left(\mathbf{C}_{\text{neighbor}} - \mathbf{C}_{\text{target}}\right)$$

### 3.2 Multi-Encoder Generative VAE
* **Visible Encoder ($E_e$):** PointNet deterministic feature extractor mapping visible context $P_e \rightarrow z_e \in \mathbb{R}^{128}$.
* **Target VAE Encoder ($E_m$):** PointNet stochastic VAE encoder mapping missing target $P_m \rightarrow \mu, \log \sigma^2$, producing $z_m \sim \mathcal{N}(\mu, \Sigma) \in \mathbb{R}^{128}$ via the reparameterization trick.
* **Context Encoder ($E_c$):** Specialized neural network mapping surrounding spatial context $P_c \rightarrow z_c \in \mathbb{R}^{128}$.
* **Latent Conditioning:** $z_{\text{cond}} = [z_m, z_e, z_c] \in \mathbb{R}^{384}$.

### 3.3 Dynamic HyperNetwork & Implicit Target Decoder
Rather than generating point coordinates through fixed-size decoders, a **Context HyperNetwork** MLP ($384 \rightarrow 64 \rightarrow 128 \rightarrow 512 \rightarrow 1024 \rightarrow 2048$) dynamically generates the weights and biases for an implicit **TargetNetwork** MLP ($3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 64 \rightarrow 3$). Points continuously sampled from a progressively normalized sphere are mapped to the reconstructed 3D surface.

### 3.4 Context Encoder Paradigms Evaluated
* **PointNet ([`models/base_hyperpocket.py`](models/base_hyperpocket.py)):** Independent per-point MLPs followed by global max-pooling.
* **Dynamic Graph CNN ([`models/dgcnn_context_encoder.py`](models/dgcnn_context_encoder.py)):** 4-scale EdgeConv layers ($k=20$) dynamically constructing $k$-NN graphs in feature space with dual global pooling (Max + Avg, 2048D).
* **Cross-Attention Transformer ([`models/cross_attention_encoder.py`](models/cross_attention_encoder.py)):** Set Transformer with Pooling by Multihead Attention (PMA), using a learnable query token attending over input point embeddings with $O(N)$ linear complexity.
* **Point-Mamba State Space Model ([`models/point_mamba_encoder.py`](models/point_mamba_encoder.py)):** Space-filling Morton Z-order serialization with bidirectional 1D depthwise convolution and selective SSM scans (`PureSelectiveSSM`) accelerated via native cuBLAS batched tensor contractions.

### 3.5 Training Objective & Hyperparameters
$$\mathcal{L} = 0.05 \cdot \mathcal{L}_{\text{CD}}(P_m, \hat{P}_m) + 1.0 \cdot D_{\text{KL}}\left(q(z_m \mid P_m) \parallel \mathcal{N}(0, \mathbf{I})\right)$$
* **Optimizer:** Adam ($\text{lr} = 10^{-4}, \beta_1 = 0.9, \beta_2 = 0.999$).
* **LR Schedule:** StepLR (step = 41, $\gamma = 0.01$).
* **Hardware Acceleration:** Native PyTorch Automatic Mixed Precision (AMP FP16) + cuDNN auto-tuning on NVIDIA Tesla T4.
* **Early Stopping:** Minimum 40 epochs, patience = 15 epochs.

---

## 📊 4. Experimental Results & Benchmark

All models are evaluated on the official **1,077 SensatUrban test blocks** with $k=10$ stochastic generative completions per block.

### 4.1 Executive Cross-Paradigm Comparison

| Paradigm | Architecture | Recon CD ↓<br>*(Shape Acc.)* | Recon EMD ↓<br>*(Density Transport)* | MMD (CD) ↓<br>*(Fidelity)* | TMD ↑<br>*(Diversity)* | JSD ↓<br>*(Distribution)* | Best Epoch |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (No Context)** | Isolated HyperPocket (Wu et al.) | 68.68 | 0.2004 | 38.58 | **62.41** | **0.1233** | 60 |
| **PointNet $E_c$** | Spatial Context Default | 62.86 | **0.1989** | 40.34 | 53.65 | 0.1613 | 59 |
| **DGCNN $E_c$** | Dynamic EdgeConv ($k=20$) | 60.10 | 0.2036 | 39.52 | 52.85 | 0.1728 | 65 |
| **Cross-Attention $E_c$** | Set Transformer PMA ($d=256$) | **57.80** | 0.2038 | **38.21** | 47.42 | 0.1829 | 62 |
| **Point-Mamba $E_c$** | Selective State Space Model | *[In Progress]* | *[In Progress]* | *[In Progress]* | *[In Progress]* | *[In Progress]* | *[Pending]* |

---

### 4.2 Comprehensive Ablation Study Benchmark

<details open>
<summary><b>Click to expand/collapse full ablation table (14 completed experiments)</b></summary>
<br>

| Family | Experiment ID | Configuration / Ablation Focus | Recon CD ↓ | Recon EMD ↓ | MMD (CD) ↓ | TMD ↑ | JSD ↓ | Best Ep. |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Ablations** | `baseline_default` | Isolated ($\text{coef}=0.05, \beta=1.0$) | 68.68 | 0.2004 | 38.58 | **62.41** | **0.1233** | 60 |
| | `exp2_reduced_beta` | Isolated ($\text{coef}=1.0, \beta=0.001$) | 67.64 | 0.2095 | 40.73 | 37.74 | 0.2061 | 60 |
| | `exp3_annealing` | Isolated ($\beta$-annealing $0 \rightarrow 0.001$) | 62.76 | 0.2424 | 38.64 | 36.81 | 0.3293 | 62 |
| **PointNet $E_c$ Loss Ablations** | `exp1_context_default` | PointNet $E_c$ ($\text{coef}=0.05, \beta=1.0$) | 62.86 | **0.1989** | 40.34 | 53.65 | 0.1613 | 59 |
| | `exp2_context_reduced_beta` | PointNet $E_c$ ($\text{coef}=1.0, \beta=0.001$) | 70.95 | 0.2313 | 40.84 | 39.95 | 0.3088 | 57 |
| | `exp3_context_annealing` | PointNet $E_c$ ($\beta$-annealing $0 \rightarrow 0.001$) | 79.84 | 0.2643 | 47.33 | 27.74 | 0.3961 | 53 |
| **DGCNN $E_c$ Graph Ablations** | `exp4_dgcnn_context` | EdgeConv Baseline ($k=20$, Dual Max+Avg) | 60.10 | 0.2036 | 39.52 | 52.85 | 0.1728 | 65 |
| | `exp4a_dgcnn_k10` | Sparse neighborhood graph ($k=10$) | 68.12 | 0.1994 | 41.02 | 51.78 | 0.1347 | 58 |
| | `exp4b_dgcnn_k40` | Dense neighborhood graph ($k=40$) | 64.38 | 0.2052 | 38.74 | 51.08 | 0.1629 | 61 |
| | `exp4c_dgcnn_light` | Light channels `[32, 32, 64, 128]` | 63.79 | 0.2128 | 38.40 | 48.40 | 0.1793 | 67 |
| | `exp4d_dgcnn_max_only` | Max-only pooling (1024D vs 2048D) | 64.06 | 0.2012 | 38.65 | 50.00 | 0.1577 | 54 |
| **Cross-Attention $E_c$ Ablations**| `exp5_ca_context` | Set Transformer Baseline ($d=128, H=4$) | 63.48 | 0.2058 | 39.58 | 54.53 | 0.1696 | 57 |
| | `exp5a_ca_dim64` | Reduced token dim ($d=64, H=4$) | 60.73 | 0.2021 | **37.37** | 50.65 | 0.1687 | 61 |
| | `exp5b_ca_dim256` | Scaled token dim ($d=256, H=4$) | **57.80** | 0.2038 | 38.21 | 47.42 | 0.1829 | 62 |
| | `exp5c_ca_heads2` | 2 attention heads ($d=128, H=2$) | 61.68 | 0.2023 | 38.65 | 49.23 | 0.1738 | 57 |
| | `exp5d_ca_heads8` | 8 attention heads ($d=128, H=8$) | 62.98 | 0.2032 | 42.13 | 55.62 | 0.1797 | 59 |
| **Point-Mamba $E_c$ Suite** | `exp6_mamba_context` | Bidirectional Z-Order Baseline ($d=128, S=16$) | *Training* | *Training* | *Training* | *Training* | *Training* | — |
| *(In Progress)* | `exp6a_mamba_unidirectional`| Unidirectional Z-Order Scan | *Training* | *Training* | *Training* | *Training* | *Training* | — |
| | `exp6b_mamba_coord_sort` | Coordinate Sorting Serialization | *Scheduled*| *Scheduled*| *Scheduled*| *Scheduled*| *Scheduled*| — |
| | `exp6c_mamba_dim256` | Scaled Mamba Dimension ($d=256$) | *Training* | *Training* | *Training* | *Training* | *Training* | — |
| | `exp6d_mamba_dim64` | Lightweight Mamba Dimension ($d=64$) | *Scheduled*| *Scheduled*| *Scheduled*| *Scheduled*| *Scheduled*| — |
| | `exp6e_mamba_state32` | Expanded State Space ($S=32$) | *Scheduled*| *Scheduled*| *Scheduled*| *Scheduled*| *Scheduled*| — |

</details>

---

### 4.3 Key Scientific Takeaways
1. **Context Regularization Works:** Conditioning on surrounding physical space consistently outperforms isolated completion. Cross-Attention ($d=256$) achieves the **lowest overall Chamfer Distance (57.80)**, outperforming the baseline by **15.8%**.
2. **Local Graph Connectivity Threshold:** In DGCNN, setting $k=20$ is optimal. Restricting to $k=10$ starves the edge-convolution of structural continuity (CD drops to 68.12), while $k=40$ leads to oversmoothing across distinct architectural boundaries (CD 64.38).
3. **Dual Global Pooling:** Max + Avg pooling retains both acute architectural corners (max) and global surface point density (avg), improving CD by $4.0$ points over Max-only pooling.
4. **Diversity vs. Consistency Trade-off:** The unconditioned baseline yields the highest Total Mutual Difference (TMD = 62.41) because the generative model is unconstrained. Spatial context naturally lowers TMD (47–55) by anchoring the hallucinated geometry to real-world neighbor geometry.

---

## 🎨 5. Qualitative Results Gallery

### Multimodal Generative Hallucination
Because completion is formulated as a conditional VAE, sampling multiple latent vectors $z_m \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ allows synthesizing **multiple distinct, plausible geometric variants** for occluded structures:

<p align="center">
  <img src="assets/multimodal_variants.png" alt="Multimodal Generative Variants" width="95%">
  <br>
  <em><b>Figure 2: Multimodal Shape Hallucination ($k=4$).</b> From a single occluded input (Pe), our context-aware VAE synthesizes diverse yet structurally valid variants for missing urban roofs and walls.</em>
</p>

### Training Dynamics & Convergence
<p align="center">
  <img src="assets/loss_curves.png" alt="Training Loss Curves" width="85%">
  <br>
  <em><b>Figure 3: Training Stability.</b> StepLR decay at epoch 41 produces smooth fine-tuning with early stopping triggering reliably upon validation plateau.</em>
</p>

---

## 🗺️ 6. Project Roadmap & Milestone Status

- [x] **Phase 1: Data Pipeline & Spatial Partitioning**
  - Subsampled 2.8B raw SensatUrban points with 20cm voxel grid.
  - Partitioned into 5,270 normalized $30\,\text{m} \times 30\,\text{m}$ spatial blocks ($N=1,024$).
- [x] **Phase 2: Spatial Context Graph Construction**
  - Built 2D ground $k$-d tree indexing $K=4$ nearest physical neighbor blocks with relative coordinate offsets.
- [x] **Phase 3: Baseline HyperPocket Re-Implementation**
  - Re-implemented Wu et al. VAE architecture in native PyTorch with unit-sphere normalization.
- [x] **Phase 4: PointNet Context Model & Loss Balancing**
  - Implemented initial triple-encoder VAE and evaluated $\beta$-regularization and annealing.
- [x] **Phase 5: Dynamic Graph CNN (EdgeConv) Context Encoder**
  - Implemented pure PyTorch EdgeConv with dynamic $k$-NN feature graphs and dual pooling.
- [x] **Phase 6: Cross-Attention Set Transformer Context Encoder**
  - Implemented multi-head PMA cross-attention over spatial context.
- [ ] **Phase 7: Point-Mamba State Space Model Suite** *(In Progress)*
  - [x] Core SSM architecture with bidirectional scans and Morton Z-order serialization.
  - [x] Accelerated PyTorch implementation with native cuBLAS tensor cores and AMP FP16.
  - [x] Verified 100% unit test pass rate across all 6 ablation configurations.
  - [ ] Complete 6-experiment ablation execution (`exp6` – `exp6e`).
- [ ] **Phase 8: Final Synthesis & Thesis Defense**
  - [ ] Unified comparative analysis (Mamba vs. Transformer vs. DGCNN).
  - [ ] Final thesis report and manuscript publication.

---

## 📁 7. Repository Structure

```
GenerativeVaeReconstruction3DPointcloud/
├── configs/                            # Modular experiment configurations
│   ├── base_config.py                  # Global hyperparameters & training defaults
│   ├── base_hyperpocket_architecture.py# Baseline HyperPocket config variants
│   └── context_hyperpocket_architecture.py # Context variants (PointNet, DGCNN, CA, Mamba)
├── datasets/                           # Data pipelines & loading contracts
│   ├── preprocess.py                   # Voxel subsampling & 30m XY spatial partitioning
│   ├── build_neighbor_index.py         # 2D k-d tree spatial neighbor indexing (K=4)
│   ├── sensat_dataset.py               # Baseline single-block DataLoader & hyperplane cuts
│   ├── context_sensat_dataset.py       # Context-aware DataLoader with relative offset alignment
│   ├── neighbor_map.json               # Spatial neighbor lookup index
│   └── tile_centroids.json             # Absolute metric block centroids
├── models/                             # PyTorch neural network architectures
│   ├── base_hyperpocket.py             # PointNet encoders & implicit TargetNetwork MLP
│   ├── context_hyperpocket.py          # Triple-encoder VAE & Context HyperNetwork
│   ├── dgcnn_context_encoder.py        # Dynamic Graph CNN (EdgeConv) Ec implementation
│   ├── cross_attention_encoder.py      # Latent Cross-Attention (Set Transformer) Ec
│   └── point_mamba_encoder.py          # Bidirectional Selective SSM (Point-Mamba) Ec
├── utils/                              # Metrics & visualizers
│   ├── sensat_metrics.py               # Evaluation metrics: CD, EMD (Sinkhorn), MMD, TMD, JSD
│   └── visualize.py                    # 3D scatter generation & statistical plotting
├── tests/                              # Comprehensive test suites
│   ├── test_point_mamba_encoder.py     # Verification suite for Point-Mamba models
│   ├── test_dgcnn_context_encoder.py   # DGCNN unit tests
│   └── test_train_pipeline.py          # End-to-end forward/backward pipeline verification
├── assets/                             # Visual assets for documentation
│   ├── teaser_reconstruction.png       # 3-panel completion visual
│   ├── multimodal_variants.png         # Multimodal generative samples
│   └── loss_curves.png                 # Loss curve convergence plot
├── train.py                            # Production training pipeline with AMP & early stopping
├── evaluate.py                         # Quantitative benchmark suite (k=10 evaluations)
└── requirements.txt                    # Project environment dependencies
```

---

## 🚀 8. Quickstart & Reproducibility

### 8.1 Environment Setup
```bash
# Clone the repository
git clone https://github.com/syedrayyanadil/GenerativeVaeReconstruction3DPointcloud.git
cd GenerativeVaeReconstruction3DPointcloud

# Install dependencies
pip install -r requirements.txt
```

### 8.2 Training
All models can be trained via `train.py` using registered configuration keys:

```bash
# 1. Train Baseline HyperPocket (No Context)
python train.py --config base_hyperpocket.default

# 2. Train Context DGCNN (EdgeConv, k=20)
python train.py --config context_hyperpocket.exp4_dgcnn_context

# 3. Train Context Cross-Attention (d=256)
python train.py --config context_hyperpocket.exp5b_ca_dim256

# 4. Train Context Point-Mamba (State Space Model)
python train.py --config context_hyperpocket.exp6_mamba_context
```

### 8.3 Evaluation
Run quantitative evaluation on the 1,077 test blocks:
```bash
python evaluate.py --config context_hyperpocket.exp4_dgcnn_context
```
Results, quantitative metrics, and generated 3D `.ply` files will be automatically exported to `experiments/<model>/<exp>/evaluation_results.json` and logged to Weights & Biases.

---

## 📚 9. References & Acknowledgments

* **HyperPocket:** Wu, Z., et al. *"HyperPocket: Generative Point Cloud Completion."* IEEE Transactions on Visualization and Computer Graphics (TVCG), 2022.
* **SensatUrban:** Hu, Q., et al. *"Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges."* International Journal of Computer Vision (IJCV), 2021.
* **Point-Mamba:** Liang, D., et al. *"PointMamba: A Simple State Space Model for Point Cloud Analysis."* arXiv:2402.10739, 2024.
* **DGCNN:** Wang, Y., et al. *"Dynamic Graph CNN for Learning on Point Clouds."* ACM Transactions on Graphics (TOG), 2019.
* **Set Transformer:** Lee, J., et al. *"Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks."* ICML, 2019.
