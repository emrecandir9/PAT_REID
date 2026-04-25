# PAT Urban ReID — Comprehensive Improvement Analysis

## 1. Understanding Your Task & Data

### The Dataset: UrbanElementsReID
- **1,088 identities**, **11,175 images**, **3 cameras**
- **~10 images per identity** (very sparse — much sparser than Market-1501 which has ~17/identity)
- **Urban elements** = benches, lampposts, bollards, signs, trash bins, etc. — NOT people
- Query = Gallery = Train split (self-evaluation during training)
- Real test has separate query/gallery with unknown IDs

### Current Pipeline
| Component | Current Setting |
|---|---|
| Backbone | DeiT-Base (ViT-Base arch, distilled weights) |
| Part Attention | 3 part tokens + masked attention (designed for human body parts) |
| Input | 256×128 (2:1 portrait ratio, designed for pedestrians) |
| Loss | Label-smooth CE + Soft Triplet + PEDAL clustering |
| Augmentation | Flip + Pad/Crop + LGT (Local Grayscale) |
| Re-ranking | k-reciprocal (k1=50, k2=15, λ=0.3) |
| Optimizer | SGD, LR=0.001, linear warmup 5 epochs, 40 total |
| Batch | 64 images, softmax_triplet sampler, 4 instances/ID |

---

## 2. Fundamental Mismatch Analysis

> [!IMPORTANT]
> The PAT model was designed for **pedestrian** ReID. Several core design choices are suboptimal for urban elements.

### 2.1 Part Attention — The Key Problem

PAT splits the image into **3 horizontal part tokens** using a fixed mask:
```
Part 1: top half     (rows 0–7)   → designed for "head/shoulders"
Part 2: middle half  (rows 4–11)  → designed for "torso"
Part 3: bottom half  (rows 8–15)  → designed for "legs/feet"
```

**For pedestrians**: This is brilliant — head, torso, legs have distinct discriminative features.

**For urban elements**: This is wrong. A bollard, a bench, or a sign doesn't have a "head" and "legs." The fixed 3-part horizontal split:
- Wastes capacity on semantically meaningless regions
- Misses the actual discriminative parts (logo on a bench, texture pattern, mounting bracket shape)
- Assumes vertical orientation (pedestrians stand upright), but urban elements are often wider than tall

### 2.2 Aspect Ratio — 256×128 is Wrong

The 256×128 (2:1 portrait) ratio is designed for pedestrians who are **taller than wide**. Urban elements are often:
- **Square**: signs, trash bins, electrical boxes
- **Landscape**: benches, railings, barriers
- **Variable**: lampposts (portrait), bollards (square crop), fences (landscape)

Forcing everything into 2:1 portrait distorts many elements.

### 2.3 Augmentations — Too Conservative

Only horizontal flip + padding + LGT are active. Missing critical augmentations for urban ReID:
- **No Random Erasing** — urban elements get occluded by pedestrians, vehicles, other objects
- **No Color Jitter** — urban elements are photographed under very different lighting (sun/shadow/night)
- **No vertical flip** — some urban elements look similar upside-down (symmetric designs)

---

## 3. Data-Centric Improvements

### 📊 Priority: HIGH

#### 3.1 Aspect Ratio Experiments

| Experiment | Size | Patches (y×x) | Ratio | Why |
|---|---|---|---|---|
| Current | 256×128 | 16×8=128 | 2:1 portrait | Pedestrian default |
| **Square** | **224×224** | **14×14=196** | **1:1** | Natural for objects |
| Landscape | 128×256 | 8×16=128 | 1:2 | For wide elements |
| Large square | 256×256 | 16×16=256 | 1:1 | More spatial info, slower |

> [!TIP]
> **224×224 is the strongest candidate** — matches ImageNet pretraining resolution, produces more patches for the attention mechanism, and doesn't distort any element type.

To try this, change in YAML:
```yaml
INPUT:
  SIZE_TRAIN: [224, 224]
  SIZE_TEST: [224, 224]
MODEL:
  STRIDE_SIZE: [16, 16]
```

#### 3.2 Enable Random Erasing (REA)

Simulates occlusion — critical for urban scenes where elements are partially hidden:
```yaml
INPUT:
  REA:
    ENABLED: True
    PROB: 0.5
```

#### 3.3 Enable Color Jitter

Urban elements experience dramatic lighting changes across cameras:
```yaml
INPUT:
  CJ:
    ENABLED: True
    PROB: 0.5
    BRIGHTNESS: 0.2
    CONTRAST: 0.2
    SATURATION: 0.15
    HUE: 0.1
```

#### 3.4 Enable Random Patch (RPT)

Random Patch replaces random regions with patches from other images — strong regularization:
```yaml
INPUT:
  RPT:
    ENABLED: True
    PROB: 0.3
```

#### 3.5 Test-Time Augmentation (TTA)

In `update.py`, the extract_feature function already does a 2-pass average. Extend this to:
- **Horizontal flip** averaging (standard in ReID)
- **Multi-scale** testing at 2-3 resolutions

This is **free accuracy** at inference time.

#### 3.6 Super-Resolution Preprocessing

You already have Real-ESRGAN integrated from a previous session. For very small crops (< 64px on any side), upscaling 2× before resizing could recover discriminative texture details that bilinear interpolation destroys.

---

## 4. Model-Centric Improvements

### 🏗️ Architectural Changes

#### 4.1 Disable Part Attention → Use Plain ViT

Since part attention assumes human body parts, **plain ViT may outperform PAT for urban elements**:

Change in YAML:
```yaml
MODEL:
  NAME: 'vit'  # instead of 'part_attention_vit'
```

This removes the 3 part tokens and PEDAL clustering loss entirely. The model uses only the CLS token for global representation. **This is a very important baseline to establish.**

#### 4.2 Learnable Part Tokens (If Keeping PAT)

Instead of the fixed 3-part horizontal mask, the part tokens should learn **where to attend** from the data. The current mask is hardcoded in `attn_mask_generate`:
```python
mask_ |= generate_2d_mask(H,W,0,0,W,H/2,1,False, device)      # top half
mask_ |= generate_2d_mask(H,W,0,H/4,W,H/2,2, False, device)   # middle
mask_ |= generate_2d_mask(H,W,0,H/2,W,H/2,3, False, device)   # bottom half
```

**Modification idea**: Remove the mask entirely and let part tokens learn to attend freely via standard cross-attention. This requires modifying `part_Attention_ViT.forward_features()` to pass `mask=None` and adjusting `part_Attention.forward()` to handle no mask.

#### 4.3 Different Number of Part Tokens

Currently fixed at 3. Urban elements might benefit from:
- **1 part token** (for small, simple elements like bollards)
- **4–6 part tokens** (for complex, multi-component elements like benches with armrests, seat, backrest, legs)

This requires modifying `part_Attention_ViT.__init__()` to be configurable.

#### 4.4 Backbone Alternatives for Kaggle T4

| Backbone | Params | embed_dim | Pretrained | Notes |
|---|---|---|---|---|
| `deit_base_patch16_224` | 87M | 768 | ✅ DeiT distilled | **Current** |
| `vit_base_patch16_224` | 87M | 768 | ✅ ImageNet-21k | Tried before, good baseline |
| `deit_small_patch16_224` | 22M | 384 | ✅ DeiT distilled | Faster, enables larger batch |
| `vit_small_patch16_224` | 28M | 768 (8 layers) | ✅ ImageNet | Trade quality for speed |
| **CLIP ViT-B/16** | 87M | 768 | ✅ CLIP | **Strong zero-shot features** (needs custom loading) |
| **DINOv2 ViT-B/14** | 87M | 768 | ✅ Self-supervised | **Best general visual features** (needs custom loading) |

> [!IMPORTANT]
> **DINOv2** and **CLIP** pretrained weights are significantly better than ImageNet for object-level discrimination because they're trained on much more diverse data with stronger feature learning objectives. These would require modifying the weight loading code but are very promising.

### 🎯 Loss Function Improvements

#### 4.5 Circle Loss (Replace Triplet)

Circle Loss (CVPR 2020) is the state-of-the-art metric learning loss for ReID. It provides better convergence than triplet loss by weighting each positive/negative pair individually based on their current optimization status.

```python
class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=256):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, feats, labels):
        # Cosine similarity
        sim_mat = torch.matmul(feats, feats.T)
        # ... (similarity-to-loss conversion)
```

#### 4.6 ArcFace Loss (Replace Label-Smooth CE)

ArcFace adds angular margin to the softmax classification, creating tighter clusters:
```yaml
MODEL:
  COS_LAYER: True
  ID_LOSS_TYPE: 'arcface'
```

The code already has `arcface.py` in the loss directory — it just needs to be wired in.

#### 4.7 Center Loss (Already Implemented)

Center loss pulls features toward their class center. Can be enabled:
```yaml
MODEL:
  IF_WITH_CENTER: 'yes'
  METRIC_LOSS_TYPE: 'triplet_center'
```

#### 4.8 Loss Weight Tuning

Currently both losses have weight 1.0. Worth experimenting:
```yaml
MODEL:
  ID_LOSS_WEIGHT: 1.0
  TRIPLET_LOSS_WEIGHT: 1.0  # try 0.5, 1.5, 2.0
```

### ⚙️ Training Hyperparameters

#### 4.9 Optimizer: AdamW vs SGD

Current: SGD with LR=0.001. ViT models typically train better with **AdamW**:
```yaml
SOLVER:
  OPTIMIZER_NAME: 'Adam'  # uses AdamW internally
  BASE_LR: 0.00035        # Adam needs lower LR
  WEIGHT_DECAY: 0.05      # higher WD for transformers
```

#### 4.10 Learning Rate Schedule

Current: Linear warmup + fixed. Better options:
- **Cosine annealing** — smooth decay, usually better for ViTs
- **Warmup to higher LR** then cosine — standard ViT recipe

#### 4.11 Longer Training

40 epochs may not be enough for 1,088 classes. Try:
```yaml
SOLVER:
  MAX_EPOCHS: 80  # or 120
  WARMUP_EPOCHS: 10
```

#### 4.12 Larger Batch Size with Gradient Accumulation

Triplet loss benefits greatly from **larger batches** (more negatives to contrast). If batch 64 doesn't OOM, try 128 with gradient accumulation.

#### 4.13 NUM_INSTANCE Tuning

Currently 4 instances per ID. With 1,088 IDs and batch 64, you get 64/4=16 IDs per batch. Try:
- `NUM_INSTANCE: 8` (fewer IDs but more positives → better triplet mining)
- `NUM_INSTANCE: 2` (more IDs → more diverse negatives)

#### 4.14 Unfreeze Patch Embeddings

Currently `FREEZE_PATCH_EMBED: True`. For a domain this different from ImageNet, unfreezing should be tried:
```yaml
MODEL:
  FREEZE_PATCH_EMBED: False
```

---

## 5. Ensemble Strategies

### 5.1 Feature Concatenation Ensemble

Train 2-3 models with different configs, extract features, concatenate at test time:

```python
feat = torch.cat([
    model_vit(img),           # 768-d
    model_pat(img),           # 768-d
    model_deit_small(img),    # 384-d
], dim=1)  # 1920-d total
```

Then run re-ranking on the concatenated feature. **This is the strongest ensemble approach for ReID.**

### 5.2 Distance Matrix Averaging

Simpler than feature concat — just average the distance matrices:

```python
distmat_final = 0.5 * distmat_model1 + 0.5 * distmat_model2
```

### 5.3 Diverse Model Ensemble Ideas

| Model A | Model B | Diversity Source |
|---|---|---|
| PAT (ViT-B) | Plain ViT (ViT-B) | Part attention vs global |
| ViT-B @ 256×128 | ViT-B @ 224×224 | Aspect ratio |
| DeiT-Base | ViT-Base | Different pretraining |
| ViT-B with REA | ViT-B without REA | Augmentation diversity |
| ViT-B epoch 30 | ViT-B epoch 40 | Temporal ensemble (snapshot ensemble) |

### 5.4 Query Expansion (QE)

After initial ranking, **average query feature with top-K gallery features** and re-query. This is applied post-hoc and doesn't require retraining:

```python
# Average Query Expansion
alpha = 3  # top-K
for i in range(num_query):
    top_k_idx = np.argsort(distmat[i])[:alpha]
    qf[i] = (qf[i] + gf[top_k_idx].mean(0)) / 2
# Re-compute distances with enriched queries
```

---

## 6. Inference/Post-Processing Improvements

### 6.1 Re-Ranking Parameter Tuning

Current: k1=50, k2=15, λ=0.3.

These need tuning! For a dataset of 11K images:
| Parameter | Current | Try |
|---|---|---|
| k1 | 50 | 20 (original paper default), 30, 40 |
| k2 | 15 | 6 (original paper default), 10 |
| λ | 0.3 | 0.3 (keep), 0.4, 0.5 |

> [!TIP]
> k1=50 is very aggressive for 11K images. The original paper uses k1=20 on Market-1501 (19K images). Try k1=20–30 first.

### 6.2 Feature Normalization at Test Time

Currently `FEAT_NORM: True` and `NECK_FEAT: 'before'`. Experiment with:
```yaml
TEST:
  NECK_FEAT: 'after'   # use BN-normalized feature
```

The "after" feature often works better with Euclidean distance, while "before" is better with cosine.

---

## 7. Creative / Advanced Ideas

### 7.1 Class-Conditional Training

Since your dataset has class labels (bench, lamppost, etc.), you could:
1. **Add a class classifier head** alongside the ID head
2. Jointly train class classification + ID discrimination
3. The class features would force the model to learn class-specific representations

### 7.2 Camera-Aware Training

With 3 cameras, you can enable camera-domain adaptation:
```yaml
DATALOADER:
  CAMERA_TO_DOMAIN: True
```

This treats each camera as a domain, helping the model generalize across viewpoints.

### 7.3 Self-Supervised Pre-Training on Your Data

Before supervised training, do a few epochs of DINO-style self-supervised learning on your urban images (no labels needed). This adapts the ViT to the visual statistics of your specific urban environment.

### 7.4 Multi-Crop Training

Instead of using a single crop at 256×128, train with multiple random crops at different scales from the same image, similar to the DINO multi-crop strategy.

### 7.5 Knowledge Distillation

Train a large model (ViT-Large) on a high-memory machine, then distill its knowledge into your ViT-Base for Kaggle deployment.

---

## 8. Prioritized Experiment Plan

Ordered by **expected impact × feasibility on Kaggle T4**:

### Tier 1 — High Impact, Easy to Implement
| # | Experiment | Change | Expected Impact |
|---|---|---|---|
| 1 | **Square input 224×224** | YAML only | 🟢🟢🟢 Major — removes distortion |
| 2 | **Enable REA + Color Jitter** | YAML only | 🟢🟢 Strong regularization |
| 3 | **Plain ViT baseline** (`NAME: 'vit'`) | YAML only | 🟢🟢 Removes wrong part priors |
| 4 | **Re-ranking k1=20, k2=6** | update.py | 🟢🟢 Original paper defaults |
| 5 | **TTA: horizontal flip** | update.py | 🟢 Free accuracy |

### Tier 2 — Medium Impact, Moderate Effort
| # | Experiment | Change | Expected Impact |
|---|---|---|---|
| 6 | Unfreeze patch embed | YAML only | 🟡 Better domain adaptation |
| 7 | AdamW optimizer + cosine schedule | YAML + code | 🟡 Better convergence |
| 8 | Longer training (80+ epochs) | YAML only | 🟡 More training |
| 9 | ArcFace loss | YAML + wiring | 🟡 Tighter clusters |
| 10 | NUM_INSTANCE: 8 | YAML only | 🟡 Better triplet mining |

### Tier 3 — High Impact, More Effort
| # | Experiment | Change | Expected Impact |
|---|---|---|---|
| 11 | Feature concat ensemble (2 models) | update.py | 🟢🟢🟢 Best for competition |
| 12 | Query Expansion | update.py | 🟢🟢 Post-hoc boost |
| 13 | CLIP/DINOv2 backbone | model code | 🟢🟢🟢 Best pretrained features |
| 14 | Remove fixed part mask | model code | 🟢🟢 Learnable parts |
| 15 | Circle Loss | loss code | 🟢 Better metric learning |

> [!IMPORTANT]
> **Start with Tier 1 experiments.** They require only YAML changes and address the most fundamental mismatches (aspect ratio, augmentation, part attention suitability). Each can be tested independently in ~6–7 hours on Kaggle T4.

---

## 9. Recommended First Experiment

The single highest-impact change is a combination of:
```yaml
MODEL:
  NAME: 'vit'    # Plain ViT, no part attention
INPUT:
  SIZE_TRAIN: [224, 224]
  SIZE_TEST: [224, 224]
  REA:
    ENABLED: True
  CJ:
    ENABLED: True
```

This addresses the three biggest issues (wrong parts, wrong ratio, weak augmentation) in one run. Compare mAP/Rank-1 against your current PAT baseline to see the delta.
