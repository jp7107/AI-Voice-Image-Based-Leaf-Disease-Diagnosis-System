# AI + Voice + Image Based Leaf Disease Diagnosis System
## End-to-End Engineering + ML Research Plan

> **Status:** Living design document. **All research is now web-verified** — datasets, exact model IDs/licenses, the weather API, and every paper citation carry primary-source links so that **no dataset, model, API, or paper is fabricated**. Facts not confirmable from a primary source are explicitly marked **UNVERIFIED**.

> **Labeled assumptions (correct me if wrong):**
> - **A1.** This is an academic / final-year-style ML research prototype (advisor: "Manish Sir"), to be *implemented, trained, evaluated, demonstrated, and defended* — not a funded production product.
> - **A2.** Hardware available is *moderate*: a laptop (possibly Apple Silicon), plus optional free/cheap cloud GPU (Google Colab / Kaggle / a rented consumer GPU). No guaranteed multi-GPU cluster.
> - **A3.** Timeline is on the order of **weeks**, single developer (you), possibly with the advisor reviewing milestones.
> - **A4.** The scientific goal is a *defensible* demonstration that multimodal + prototype/few-shot learning is (or is not) worthwhile for leaf disease diagnosis — an honest result beats an inflated one.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Analysis (critical, component-by-component)](#2-architecture-analysis)
3. [Recommended Technology Stack](#3-recommended-technology-stack)
4. [Dataset Research](#4-dataset-research)
5. [Selected Dataset(s)](#5-selected-datasets)
6. [Dataset Preparation](#6-dataset-preparation)
7. [Image Pipeline](#7-image-pipeline)
8. [ViT Pipeline](#8-vit-pipeline)
9. [Voice + Whisper Pipeline](#9-voice--whisper-pipeline)
10. [Text Encoder Pipeline](#10-text-encoder-pipeline)
11. [Weather Pipeline](#11-weather-pipeline)
12. [Cross-Attention Architecture](#12-cross-attention-architecture)
13. [Prototype Memory](#13-prototype-memory)
14. [Meta-Learning / Few-Shot Architecture](#14-meta-learning--few-shot-architecture)
15. [Training Strategy](#15-training-strategy)
16. [Loss Functions](#16-loss-functions)
17. [Experiment Plan](#17-experiment-plan)
18. [Evaluation Metrics](#18-evaluation-metrics)
19. [Backend Architecture](#19-backend-architecture)
20. [Database Schema](#20-database-schema)
21. [Frontend Architecture](#21-frontend-architecture)
22. [Project Folder Structure](#22-project-folder-structure)
23. [API Design](#23-api-design)
24. [Exact Implementation Phases](#24-exact-implementation-phases)
25. [MVP Version](#25-mvp-version)
26. [Research-Grade Version](#26-research-grade-version)
27. [Testing](#27-testing)
28. [Deployment](#28-deployment)
29. [Research / Literature Review](#29-research--literature-review)
30. [Risks and Limitations](#30-risks-and-limitations)
31. [Final End-to-End Pipeline](#31-final-end-to-end-pipeline)
32. [Exact Next Steps](#32-exact-next-steps)
- [START HERE](#start-here)

---

## 1. Executive Summary

**What the system is.** A multimodal leaf-disease diagnosis service. A farmer opens a web/mobile app, captures a leaf photo, optionally speaks/types symptoms, and (optionally) shares location so weather context can be attached. The system segments the leaf, encodes the **image** (Vision Transformer), the **symptom text** (Whisper ASR → BERT/BioBERT), and the **weather** (small MLP), fuses them, and predicts a disease by **similarity to learned class prototypes**. It returns disease + confidence + supporting symptoms + weather context + curated treatment advice.

**The single most important engineering truth about this project.** No public dataset contains *image + voice + weather + label* for the same leaf. Therefore the multimodal dataset must be **constructed**: real images and labels, **derived** symptom text, **synthetic** voice (TTS of the text), and **attached** weather. This is scientifically acceptable **only if you prevent the derived/synthetic modalities from leaking the label.** If symptom text or weather is generated *from the label*, then any "multimodal improvement" is an artifact, not a result. This document treats leakage control as a first-class design constraint, not an afterthought (see [§17](#17-experiment-plan), [§30](#30-risks-and-limitations)).

**The second most important truth.** The famous lab dataset (PlantVillage) is single-leaf, uniform-background, studio images. Models trained on it reach ~99% in-distribution and then **collapse on real field photos** because they latch onto background/lighting. A credible project *must* report **cross-dataset** performance (train lab → test field). That is where your real contribution — few-shot adaptation and robustness — is demonstrated or refuted.

**Recommended build order (dependency-first, not hype-first):**

```
Image classifier baseline (ViT)  →  honest evaluation + cross-dataset test
   → add text modality (leakage-controlled)  → add weather modality
      → cross-attention fusion  → prototype classifier  → few-shot/meta-learning
         → backend API  → frontend  → deployment
```

**MVP vs research-grade.** The MVP is a *working, defensible* system: ViT + prototype classifier + Whisper→BERT text + weather MLP + simple concat/attention fusion + curated recommendations + web UI. The research-grade version adds true episodic meta-learning (Prototypical Networks), cross-attention over token sequences, cross-dataset generalization studies, calibration, and ablations. **Build the MVP fully before adding research components** — each research component is then measured as a delta over a real baseline.

**Headline component verdicts** (full reasoning in [§2](#2-architecture-analysis)):

| Component | Verdict | Note |
|---|---|---|
| ViT image encoder | **Keep** | Pretrained ViT-B/16 fine-tuned; strong, standard, justified. |
| Leaf segmentation | **Keep (MVP-light)** | Cheap background removal helps field robustness; heavy SAM is research-grade only. |
| Whisper ASR | **Keep** | `small`/`base` runs locally; symptom voice → text is genuinely useful UX. |
| BERT/BioBERT text encoder | **Keep, but pick carefully** | BioBERT is *biomedical*, not *botanical*; a general or sentence encoder may match or beat it. Test it, don't assume. |
| Weather MLP | **Keep with skepticism** | Only helps if weather genuinely correlates with disease *without leaking the label*. Must be ablated. |
| Cross-attention fusion | **Keep for research-grade; simplify for MVP** | Cross-attention is only meaningful over **token sequences**, not 3 pooled vectors. Design accordingly. |
| Prototype memory | **Keep** | Clean, defensible, enables few-shot; cheap to implement. |
| Meta-learning (few-shot) | **Keep — as Prototypical Networks** | Feasible on modest hardware; MAML is not worth the pain here. |
| Similarity matching | **Keep** | Cosine similarity to prototypes; the natural classifier for this design. |
| "Adapt to new *images*" claim | **Reframe** | Meta-learning helps new *classes* with few shots; it is not a magic robustness fix for the same classes. Say so honestly. |

---

## 2. Architecture Analysis

For each block: **(1) what it does · (2) why it's needed · (3) is it technically justified · (4) simplest practical impl · (5) research-grade impl · (6) MVP omit? · (7) assumptions.**

> Guiding principle you asked for: *prefer pretrained models + lightweight trainable layers*. Almost nothing here is trained from scratch. The trainable surface is small: projection heads, a weather MLP, a fusion module, and (optionally) light ViT fine-tuning.

### 2.1 Mobile/Web App + Input Options (image / voice / existing leaf)

1. **What:** Entry point; captures image, optional voice, optional "diagnose an existing image", plus login.
2. **Why:** It's the product surface; also where you gather GPS (for weather) and consent.
3. **Justified?** Yes, trivially.
4. **Simplest:** A single-page web app (React) with file upload + `MediaRecorder` for audio + a "use my location" button. No native app needed for a defense demo.
5. **Research-grade:** PWA / React Native for real field capture, offline queueing, camera guidance overlays.
6. **MVP omit?** Keep a thin version. Login can be a stub for the demo (see [§33 security] in requirements).
7. **Assumptions:** Browser camera/mic permissions; farmer has connectivity at least intermittently.

### 2.2 Preprocessing — Leaf Segmentation / Background Removal / Normalization

1. **What:** Validates the image, removes background, isolates the leaf, normalizes for the encoder.
2. **Why:** Background is the #1 confounder in leaf-disease models. Removing it is the cheapest robustness win and directly attacks the lab→field gap.
3. **Justified?** Strongly. This is arguably higher ROI than the fancy fusion.
4. **Simplest (MVP):** Resize + RGB + normalize, plus a **classical or lightweight background removal**: HSV/ExG green-masking + largest-contour crop, **or** the off-the-shelf `rembg` (U²-Net) model for a clean cutout. Deterministic, fast, CPU-friendly.
5. **Research-grade:** **SAM / SAM2** (Segment Anything) with a foreground point prompt, or a fine-tuned **U-Net/DeepLab** if you have masks. Report accuracy with vs without segmentation as an ablation.
6. **MVP omit?** Don't omit background removal entirely — it's cheap and it matters. You *can* omit learned segmentation and use `rembg`/classical.
7. **Assumptions:** One dominant leaf per image; if multiple leaves, MVP crops the largest.

### 2.3 ViT — Image Feature Extraction

1. **What:** Encodes the leaf image into a feature/embedding.
2. **Why:** Disease signal is visual texture/lesion morphology; a strong pretrained image encoder is the backbone of the whole system.
3. **Justified?** Yes. ViT is a reasonable, defensible choice. **Caveat:** on *small* datasets ViT can underperform CNNs unless pretrained + regularized. Pretrained ViT-B/16 (ImageNet) fine-tuned is fine; keep an **EfficientNet/ResNet** CNN baseline as a sanity check and possible stronger baseline.
4. **Simplest:** `google/vit-base-patch16-224` as a **frozen feature extractor** → 768-d embedding → train only a small head/projection. Runs on CPU for inference.
5. **Research-grade:** Fine-tune the last *k* transformer blocks; or use **DINOv2** self-supervised features (often better for transfer). Compare ViT vs CNN vs DINOv2.
6. **MVP omit?** No — this is the core. But "frozen ViT + linear head" is a legitimate MVP.
7. **Assumptions:** Input normalized to the model's expected stats; 224×224.

**Tensor contract:** `image [B,3,224,224] → ViT → patch tokens [B,197,768] → (CLS or mean-pool) → z_image [B,768] → proj → [B,D]` (we standardize on **D=256** in the fused space; see [§12](#12-cross-attention-architecture)).

### 2.4 Whisper — Speech-to-Text (ASR)

1. **What:** Transcribes spoken symptoms to text.
2. **Why:** Farmers may prefer speaking (literacy, convenience, local language). It's a real UX value-add and a legitimate multimodal input.
3. **Justified?** Yes for UX. **But be honest in the write-up:** for the *model*, voice contributes nothing beyond the text it produces — Whisper is a front-end to the text encoder, not a separate learned modality. Don't claim "voice features" as a distinct learned signal; it's ASR → text.
4. **Simplest:** `whisper base`/`small` (or `faster-whisper`) run locally/offline; 16 kHz mono; ≤30 s clips.
5. **Research-grade:** `whisper large-v3` or a fine-tuned regional-language model; measure WER on your synthetic/collected clips; handle code-switching.
6. **MVP omit?** You can ship the model path with **typed text** first and add Whisper as a thin wrapper — the text encoder doesn't care whether text came from voice or keyboard. This de-risks the pipeline.
7. **Assumptions:** Clean-ish audio; short utterances; language supported by the chosen Whisper size.

### 2.5 BERT / BioBERT — Text Encoder

1. **What:** Turns symptom text into a fixed embedding.
2. **Why:** Symptom descriptions ("yellow spots, brown edges") carry disease-discriminative information complementary to the image.
3. **Justified?** Conditionally. **BioBERT/PubMedBERT are trained on human biomedical literature, not plant pathology.** Their domain advantage here is unproven. A general `bert-base` or a **sentence-transformer** (e.g., MiniLM) may match or beat them and is lighter. **Decision: benchmark 2–3 encoders, don't assume BioBERT wins.**
4. **Simplest:** Sentence-transformer (`all-MiniLM-L6-v2`, 384-d) mean-pooled — small, fast, strong general semantics.
5. **Research-grade:** Compare `bert-base-uncased` (768), BioBERT (768), PubMedBERT (768), MiniLM (384); optionally light fine-tune on your symptom corpus.
6. **MVP omit?** The text branch can be *optional at inference* (missing-modality handling, [§30](#30-risks-and-limitations)). But include it to justify "multimodal".
7. **Assumptions — CRITICAL:** Training text must **not** be a near-deterministic function of the label, or text→label leaks and the whole multimodal claim is void. See [§6](#6-dataset-preparation)/[§17](#17-experiment-plan) for the leakage-control protocol (template diversity, paraphrase, symptom-only vocabulary, disjoint templates across splits).

### 2.6 Weather API → Weather Features

1. **What:** Fetches temperature/humidity/rainfall/wind (etc.) for the user's location/time and encodes them.
2. **Why:** Disease pressure is genuinely weather-driven (e.g., humidity/leaf-wetness favor fungal blights/mildews). In principle weather is a real prior.
3. **Justified?** In principle yes; in *this dataset reality*, **only if weather is attached from a real independent source tied to where/when the image was taken** — which lab datasets don't have. If you synthesize weather from the label, it's circular. **Decision:** treat weather as an **optional context feature**, attach it honestly (real historical weather by lat/lon/date when metadata exists; otherwise clearly label it as a *demonstration-only* input), and **ablate it**. Report its true marginal value even if ~0.
4. **Simplest:** Pull current/historical weather by lat/lon from a free API; standardize the numeric vector; small MLP → embedding.
5. **Research-grade:** Use datasets/regions where real acquisition location+date exist so weather is genuine; model leaf-wetness/degree-day features; study weather's causal contribution.
6. **MVP omit?** Can be omitted from the *model* for the first baselines and shown in the *UI* as context. Add to the model only when you can attach it without leakage.
7. **Assumptions:** Location consent; a free API tier is sufficient; missing weather is handled with a learned "missing" embedding / zero-imputation + mask.

### 2.7 Cross-Attention Fusion → Fused Representation

1. **What:** Combines image, text, weather into one representation with attention.
2. **Why:** Fusion lets one modality condition on another (e.g., text "spots" attends to lesion image tokens). This is the intended research centerpiece.
3. **Justified?** Yes for research-grade — **with a caveat you must design around:** cross-attention is only meaningful between **sequences of tokens**. If you first pool each modality to a single vector, "cross-attention" over length-1 sequences degenerates into a gated weighting (≈ concatenation + MLP). **Decision:** feed **ViT patch tokens (197×768)** and **BERT token embeddings (L×768)** into cross-attention; treat weather as a single conditioning token or FiLM modulation. That's what makes the module earn its place.
4. **Simplest (MVP):** **Concatenate** `[z_image ‖ z_text ‖ z_weather]` → LayerNorm → MLP → fused `[B,256]`. Honestly, this is a strong baseline and often within noise of fancy fusion on small data.
5. **Research-grade:** Bidirectional cross-attention (image↔text) + weather conditioning, 2–4 layers, 4–8 heads, residual+LN+FFN (a small multimodal transformer). Ablate A vs B.
6. **MVP omit?** Start with concat+MLP (Approach A). Add cross-attention (Approach B) as a measured upgrade.
7. **Assumptions:** All modalities projected to a common width; missing modalities masked.

### 2.8 Disease Prototype Memory (Learned Embeddings)

1. **What:** Stores one (or a few) embedding vector(s) per disease class — the class "prototype(s)".
2. **Why:** Enables similarity-based classification and **few-shot addition of new diseases** without retraining the network — the core adaptivity claim.
3. **Justified?** Yes, clean and defensible. Prototypes are just normalized mean embeddings of support examples.
4. **Simplest:** Compute prototype = mean of L2-normalized fused embeddings per class over the training/support set; store as a `[num_classes, D]` tensor (`.pt`/`.npy`).
5. **Research-grade:** Multiple prototypes per class (clustering), momentum/EMA prototype updates, class-balanced support sampling, optional vector DB for scale.
6. **MVP omit?** No — this *is* the classifier in this design. But you may also keep a plain softmax head as a baseline comparison.
7. **Assumptions:** Embedding space is discriminative and normalized; cosine similarity is meaningful.

### 2.9 Metric Learning

1. **What:** Shapes the embedding space so same-disease samples cluster and different diseases separate.
2. **Why:** Prototype/similarity classification only works if the space is metric-meaningful. Metric learning (contrastive/triplet/prototypical loss) is what *earns* that space.
3. **Justified?** Yes — it's the theoretical backbone that makes prototypes and few-shot work.
4. **Simplest:** Train with **prototypical loss** (episodic) or **supervised contrastive** loss on the fused embedding.
5. **Research-grade:** Combine cross-entropy (on a head) + supervised contrastive + prototypical, tune weights, study the embedding geometry (t-SNE, silhouette).
6. **MVP omit?** Can start with plain cross-entropy on a softmax head to validate the pipeline, then switch/add metric loss.
7. **Assumptions:** Enough samples per class to form stable clusters (few-shot handles the scarce ones).

### 2.10 Few-Shot Learning & 2.11 Meta-Learning

1. **What:** Learn a model / embedding that classifies a class from only K examples (few-shot), by *training across many small tasks* (meta-learning / episodic training).
2. **Why:** New/rare diseases have few labeled images. The project's headline claim ("learn from few examples, generalize to new diseases") lives here.
3. **Justified?** Yes, **but only for the "new class with few examples" claim.** It does **not** by itself make same-class predictions more robust to new backgrounds — that's a data/augmentation/segmentation problem. Keep these claims separate and honest.
4. **Simplest & recommended:** **Prototypical Networks** (Snell et al. 2017) — episodic *N-way K-shot* training; prototype = mean support embedding; classify query by nearest prototype (cosine/Euclidean). Feasible on modest hardware, no second-order gradients.
5. **Research-grade:** Compare against **Matching Networks**, **Relation Networks**, **MAML/Reptile**. Expect Prototypical Nets to be the best effort/reward. Report N-way K-shot curves (1/5/10-shot) and **novel-class** vs **known-class** performance.
6. **MVP omit?** For the very first baseline, yes — a normal classifier first. But few-shot is the *research contribution*, so it must appear in the research-grade version, done properly with episodic evaluation.
7. **Assumptions:** A pool of base classes to meta-train on; held-out novel classes for honest few-shot evaluation; **no class overlap** between meta-train and novel-test.

> **Why not MAML as the default?** MAML needs second-order gradients (or first-order approximations), is finicky to tune, and gives little advantage over Prototypical Networks on image few-shot with a strong pretrained backbone. For a defensible, reproducible student project, **Prototypical Networks** is the right call; mention MAML/Reptile as compared baselines, not the main method.

### 2.12 Similarity Matching → Disease Prediction → Confidence + Recommendations

1. **What:** Compare the query's fused embedding to all prototypes, rank by similarity, output top disease + confidence; map disease → curated advice.
2. **Why:** Turns embeddings into a decision + a safe, useful answer for the farmer.
3. **Justified?** Yes. Cosine similarity + softmax-over-similarities gives a calibrated-ish confidence and a natural "unknown" path (reject if max similarity < threshold).
4. **Simplest:** `sim = cosine(q, prototypes)`; `p = softmax(sim/τ)`; predict argmax; if `max(p) < θ` → "uncertain / consult expert".
5. **Research-grade:** Temperature calibration, open-set rejection (distance-to-nearest-prototype thresholding), top-k, per-class thresholds.
6. **MVP omit?** No — but the recommendation engine should be a **curated knowledge base keyed by disease**, *not* an LLM inventing agrochemical doses (safety). Diagnosis (ML) and advice (curated KB) are strictly separated.
7. **Assumptions:** A vetted disease→treatment table exists; confidence thresholds tuned on validation; low-confidence path preferred over overconfident wrong advice.

### 2.13 The "Meta-Learning Handles Low Data & New Images" Strip (bottom of the diagram)

**Honest reframing for your defense.** The diagram claims meta-learning yields *"accurate prediction on new/unseen images"* and robustness with low data. Precisely:
- ✅ **True:** With episodic training + prototypes, you can add a **new disease class** from a handful of examples and classify it *without retraining the backbone*.
- ✅ **True-ish:** With a good pretrained backbone + metric space, **data efficiency** improves vs training from scratch.
- ⚠️ **Overclaim risk:** "Accurate on new/unseen *images*" of *known* classes is mostly delivered by **pretraining + segmentation + augmentation + cross-dataset training**, not by meta-learning per se. In your report, attribute each gain to the right cause and **prove it with ablations** ([§17](#17-experiment-plan)) rather than asserting it.

---

## 3. Recommended Technology Stack

> Principle: pretrained models + small trainable layers; everything below runs on a laptop for inference and a single consumer/cloud GPU (or Colab/Kaggle) for training.

| Layer | Choice (MVP) | Upgrade (research-grade) | Why |
|---|---|---|---|
| Language | Python 3.10/3.11 | same | Ecosystem. |
| DL framework | **PyTorch** + torchvision | same | Standard for research; readable. |
| Model hub | **Hugging Face `transformers`** | + `timm` | Pretrained ViT/BERT/Whisper. |
| Image encoder | `transformers` ViT | `timm` ViT / DINOv2 | See [§8](#8-vit-pipeline). |
| Text encoder | `sentence-transformers` (MiniLM) | `transformers` BERT/BioBERT/PubMedBERT | See [§10](#10-text-encoder-pipeline). |
| ASR | **`faster-whisper`** (CTranslate2) or `openai-whisper` | `whisper large-v3` | CPU-friendly; offline. |
| Segmentation | **`rembg`** (U²-Net) / classical CV | **SAM / SAM2** | See [§7](#7-image-pipeline). |
| Image ops | Pillow, OpenCV, **albumentations** | same | Aug + preprocessing. |
| Few-shot | custom episodic loader (+ optional **EasyFSL**) | + compare MAML/Reptile via `learn2learn` | Keep control; avoid heavy deps. |
| Vector store | in-memory tensor `.pt` | **FAISS** or **pgvector** | Prototypes are tiny; DB only if scaling. |
| Config | YAML + `pydantic` (or Hydra) | Hydra | Reproducible runs. |
| Tracking | **TensorBoard** or **Weights & Biases** | W&B + sweeps | Curves, ablations. |
| Backend | **FastAPI** + Uvicorn | + Celery/Redis for heavy jobs | Async, typed, OpenAPI docs free. |
| DB | **SQLite** (dev) → **PostgreSQL** | + pgvector | See [§20](#20-database-schema). |
| ORM/migrations | SQLAlchemy + Alembic | same | Schema evolution. |
| Frontend | **React (Vite) + Tailwind** | React Native / PWA | Camera + `MediaRecorder` + geolocation. |
| Weather | **Open-Meteo** (`open-meteo.com`, no API key) | historical archive API (to 1940) | Verified — see [§11](#11-weather-pipeline). |
| Packaging | **Docker** + docker-compose | k8s only if needed | Reproducible deploy. |
| Testing | **pytest** + httpx | + Playwright (e2e) | See [§27](#27-testing). |

**Model sources (exact IDs, licenses, sizes — VERIFIED against Hugging Face `config.json` / official repos):**

| Role | Hugging Face ID | License | Size | Input | Output / dim |
|---|---|---|---|---|---|
| Image (MVP) | **`google/vit-base-patch16-224`** | Apache-2.0 | 86.6M | 224×224, patch16 | logits(1000) + `last_hidden_state` **[.,197,768]** |
| Image (transfer) | `google/vit-base-patch16-224-in21k` | Apache-2.0 | 86.4M | 224×224, patch16 | `last_hidden_state` (ImageNet-21k pretrain, no head) |
| Image (SSL alt) | `facebook/dinov2-base` | Apache-2.0 | 86.6M | 518 default / **patch14** (processor resizes) | features, dim **768** |
| Image (timm) | `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` | Apache-2.0 | 86.6M | 224×224 | features/logits |
| ASR | **`openai/whisper-small`** (tiny 39M / base 74M / **small 244M** / medium 769M / large-v3 1550M) | HF card Apache-2.0; OpenAI code+weights MIT | 244M | **16 kHz** mono, 30-s windows, log-Mel | text |
| ASR (fast runtime) | **`SYSTRAN/faster-whisper-*`** (CTranslate2) | MIT | same tiers | 16 kHz | text (2–4× faster, less RAM) |
| Text (MVP) | **`sentence-transformers/all-MiniLM-L6-v2`** | Apache-2.0 | ~22M *(param count UNVERIFIED)* | ≤256 wordpieces (trained 128) | embedding **384** |
| Text (general) | `bert-base-uncased` (`google-bert/bert-base-uncased`) | Apache-2.0 | 110M | ≤512 tokens | hidden **768** |
| Text (biomed) | `dmis-lab/biobert-v1.1` (BERT-base **cased**) | Apache-2.0 (repo LICENSE; HF page untagged) | ~108M | ≤512 | hidden **768** |
| Text (biomed) | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` (renamed from `...PubMedBERT...`; old ID still resolves) | MIT | ~110M | ≤512 | hidden **768** |
| Segmentation (research) | `facebook/sam-vit-*` / SAM2 (`facebook/sam2-*`) | Apache-2.0 | large | image + prompt | masks |
| Background removal (MVP) | `rembg` (U²-Net weights, downloaded on first run) | MIT (lib) | small | RGB | alpha matte |

*All licenses above are permissive (Apache-2.0/MIT) → safe for an academic prototype and open publication.* `all-MiniLM-L6-v2` param count is the only unverified figure (≈22M is the widely-cited value). Whisper license nuance: OpenAI's GitHub repo states MIT for code+weights; the mirrored Hugging Face cards tag Apache-2.0 — both permissive.

---

## 4. Dataset Research

> **VERIFIED** — every row checked via WebFetch against its primary source (arXiv / TFDS catalog / Zenodo / UCI / IEEE DataPort / GitHub / Kaggle). Fields not confirmable from a primary source are labeled **UNVERIFIED** (do not treat them as facts). WebSearch was unavailable, so discovery used the arXiv API + direct primary sources.

**Selection drivers (design-level):** we need **(a)** a large, clean, well-labeled dataset for the **image baseline + prototype space** (a lab dataset is acceptable *here only*), **(b)** a **real-field** dataset for **cross-dataset generalization testing** (the honesty check — non-negotiable), and **(c)** a **many-class / long-tail** source for **few-shot / novel-class** episodes. A lab dataset alone is *insufficient for a defensible result*.

### 4.1 Verified comparison matrix

| Dataset | Images (verified) | Classes | Crops | Field/Lab | Healthy | Masks | Text | License |
|---|---|---|---|---|---|---|---|---|
| **PlantVillage** | 54,303 (TFDS) / 54,306 (repo) | 38 | 14 | **Lab/controlled** | Yes | Yes (segmented set) | No | CC BY-NC-SA 4.0 |
| **PlantDoc** | 2,598 | 27 (≤17 disease × 13 species) | 13 | **Field** (web-scraped) | Yes | No (bbox in sep. repo) | No | CC BY 4.0 |
| **PlantWild** | >18,000 (v1) | 89 (v1) / 115 (v2) | UNVERIFIED | **Field/in-the-wild** | UNVERIFIED | No (→ PlantSeg) | **Yes** (per-disease) | CC BY-NC-ND 4.0 |
| **Cassava (2020 Kaggle)** | 21,367 labeled | 5 | 1 (cassava) | **Field** | Yes | No | No | UNVERIFIED (comp. rules) |
| **iBean / beans** | 1,295 (TFDS) / 1,296 (repo) | 3 | 1 (bean) | **Field** | Yes | No | No | MIT |
| **Plant Pathology 2020** | 3,651 | 4 | 1 (apple) | **Field** | Yes | No | No | UNVERIFIED |
| **Plant Pathology 2021** | ~23,000 (~18.6k train) | 6 (multi-label) | 1 (apple) | **Field** | Yes | No | No | UNVERIFIED |
| **UCI Rice Leaf** | 120 (40×3) | 3 | 1 (rice) | Lab/white-bg | No | No | No | CC BY 4.0 |
| **Paddy Doctor** | 16,225 | 13 (12 dis. + normal) | 1 (rice) | **Field** | Yes | No | No | CC BY-SA 4.0 |
| **COT-AD (cotton)** | >25,000 (5,000 annot.) | UNVERIFIED | 1 (cotton) | **Field + aerial** | UNVERIFIED | Yes (subset) | No | UNVERIFIED (CC on IEEE DP) |
| **DiaMOS Plant** | 3,505 (3,006 leaf + 499 fruit) | 4 leaf | 1 (pear) | **Field** | Yes (43) | No (YOLO bbox + severity) | No | CC BY 4.0 |
| **PlantSeg** | 11,400 masked + 8,000 healthy | ~115 (UNVERIFIED) | many | **Field/in-the-wild** | Yes (8,000) | **Yes (masks)** | No | UNVERIFIED |
| **FieldPlant** | 5,170 (8,629 annot. leaves) | 27 | 3 (corn/cassava/tomato) | **Field/plantation** | UNVERIFIED | No (leaf bbox) | No | CC BY 4.0 |
| **Aggregated 101-class** | UNVERIFIED | 101 | 33 | Mixed (PV+PlantDoc+PlantWild) | Mixed | No | No | UNVERIFIED |

### 4.2 Per-dataset detail (verified facts + sources)

- **PlantVillage** — GitHub `spMohanty/PlantVillage-Dataset` · Mendeley `data.mendeley.com/datasets/tywbtsjrjv/1` · TFDS `plant_village`. Papers: arXiv:1511.08060 (Hughes & Salathé), Mohanty et al. 2016 (DOI 10.3389/fpls.2016.01419). **54,303 imgs / 38 classes / 14 crops / 26 diseases.** Color, grayscale, **segmented (masks)** versions. **Lab/controlled** — single detached leaf on uniform background (its core generalization limitation). Text: no. Weather: no. **~815 MiB.** Excellent clean ViT baseline; high accuracy here does **not** imply field performance.
- **PlantDoc** — GitHub `pratikkayal/PlantDoc-Dataset` (classification "Cropped-PlantDoc"; detection is a separate bbox repo). Paper: arXiv:1911.10317 · DOI 10.1145/3371158.3371196 (CoDS-COMAD 2020). **2,598 imgs, 13 species, ≤17 disease classes (27 total).** **Field** (internet-scraped, explicitly non-lab). Overlaps PlantVillage species → the canonical **lab→field** test. Resolution/size UNVERIFIED (<1 GB).
- **PlantWild** — Paper arXiv:2408.14723 ("Snap and Diagnose") · GitHub `tqwei05/PlantWild` · HF `uqtwei2/PlantWild`. **>18,000 imgs, 89 classes (v1), 115 (v2)**, in-the-wild. **Ships per-disease text descriptions → multimodal/CLIP-ready.** License **CC BY-NC-ND 4.0** (ND = no-derivatives; check before redistributing modified data). Healthy class & crop count UNVERIFIED. Size UNVERIFIED.
- **Cassava Leaf Disease (2020)** — `kaggle.com/competitions/cassava-leaf-disease-classification` (distinct from 2019 iCassava / TFDS `cassava` = 9,430). **21,367 labeled, 5 classes (CBB, CBSD, CGM, CMD, Healthy)**, **field** (Uganda). **6.19 GB** (verified). License = competition rules (**UNVERIFIED** — confirm before commercial use). Strong single-crop field baseline; only 5 classes → weak for few-shot.
- **iBean** — GitHub `AI-Lab-Makerere/ibean` · TFDS `beans`. **1,295 imgs, 3 classes, 500×500 RGB**, field (Uganda, smartphone), **MIT**, **171.69 MiB**. No formal paper. Good smoke-test / quick transfer demo.
- **Plant Pathology 2020 (FGVC7)** — `kaggle.com/c/plant-pathology-2020-fgvc7` · arXiv:2004.11958 (Thapa et al.). **3,651 imgs, 4 classes (healthy/rust/scab/multiple), apple, field.** License UNVERIFIED.
- **Plant Pathology 2021 (FGVC8)** — `kaggle.com/c/plant-pathology-2021-fgvc8`. **~23,000 imgs (~18.6k train), 6 labels, multi-label, apple, field.** Strong for multi-label ViT. License UNVERIFIED.
- **Rice** — **UCI Rice Leaf** (`archive.ics.uci.edu/dataset/486`, DOI 10.24432/C5R013): 120 imgs, 3 classes, white-bg, **no healthy**, CC BY 4.0 — few-shot/demo only. **Paddy Doctor** (arXiv:2205.11108, DOI 10.1145/3570991.3570994; Kaggle `paddy-disease-classification`): **16,225 field imgs, 13 classes**, India, CC BY-SA 4.0, Kaggle version adds **variety+age metadata** — the recommended large rice set.
- **Cotton** — **COT-AD** (arXiv:2507.18532; IEEE DataPort DOI 10.21227/bpqn-9a12): **>25,000 imgs (5,000 annotated)**, field+aerial, DSLR. Class names/healthy/license UNVERIFIED. Smaller cotton sets exist but with weaker verifiable counts.
- **DiaMOS Plant** — Zenodo `zenodo.org/records/5557313` (DOI 10.5281/zenodo.5557313) · Agronomy 2021 (DOI 10.3390/agronomy11112107). **3,505 imgs (pear), 4 leaf classes**, field, **severity 0–4 + YOLO bbox**, CC BY 4.0, **13.1 GB**. No masks, no per-image weather (only narrative conditions).
- **PlantSeg** (masks) — arXiv:2409.04038 · GitHub `tqwei05/PlantSeg` · Zenodo `zenodo.org/records/13958858`. **11,400 masked + 8,000 healthy**, in-the-wild. **The go-to if you need segmentation masks.** Class count/license UNVERIFIED.
- **FieldPlant** — IEEE Access 2023 (DOI 10.1109/ACCESS.2023.3263042). **5,170 plantation imgs, 27 classes, 8,629 annotated leaves** (corn/cassava/tomato, Cameroon), CC BY 4.0. Excellent true-field generalization set.
- **Aggregated 101-class / 33-crop** — arXiv:2508.10817 merges PV+PlantDoc+PlantWild. A merging *recipe*, not a hosted download (count/URL UNVERIFIED).

### 4.3 License gate (⚠️ read before commercial use or redistribution)
- **Non-commercial / restricted:** PlantVillage (**NC**), PlantWild (**NC-ND** — no derivatives), Paddy Doctor (**SA** copyleft). Cassava / Plant Pathology 2020 & 2021 = **UNVERIFIED** competition rules — confirm before any commercial or redistribution use.
- **Permissive (CC BY 4.0 / MIT):** PlantDoc, UCI Rice, DiaMOS, FieldPlant, iBean (MIT). Safe for an academic prototype and open publication.

### 4.4 Number-discrepancy notes (cite carefully)
PlantVillage 54,303 (TFDS) vs 54,306 (repo); iBean 1,295 vs 1,296; **Cassava-2020 is 21,367** (often mis-cited as 21,397); TFDS `cassava` (9,430) is the **2019** set, *not* the 2020 competition. **Verified storage sizes:** Cassava 6.19 GB, DiaMOS 13.1 GB, iBean 171.69 MiB, PlantVillage ~815 MiB (all others UNVERIFIED). **Modality reality:** only **PlantSeg** (+ PlantVillage's segmented set) provides masks; only **PlantWild** provides symptom text; **no dataset provides per-image weather** — which is exactly why the weather modality must be *joined in* (see [§6](#6-dataset-preparation), [§11](#11-weather-pipeline)) and its leakage controlled.

---

## 5. Selected Dataset(s)

Selections below are **final** and tied to the verified facts in [§4](#4-dataset-research). Rationale, not just names.

| Role | Dataset | Why (verified) | Protocol constraint |
|---|---|---|---|
| **Primary** — train + prototypes | **PlantVillage** (54,303 imgs / 38 classes / 14 crops, ~815 MiB, CC BY-NC-SA) | Large, clean, balanced, converges reliably → the standard baseline & a stable prototype space. | **Lab-only.** Its accuracy is *never* the headline metric. Report it only alongside the field number below. |
| **Secondary** — cross-dataset field test | **PlantDoc** (2,598 field imgs, 27 classes, CC BY 4.0) | Web/field images that **overlap PlantVillage species** → the canonical **lab→field** generalization test; directly exposes background overfitting. | **Never used in training** in the strict protocol. Used only to measure the lab→field drop. |
| **Field stress-test** (harder, optional) | **PlantWild** (>18k, 89/115 classes) or **FieldPlant** (5,170, 27 classes, CC BY 4.0) | Larger, long-tailed, truly in-the-wild. PlantWild also ships **per-disease text**. | PlantWild is **NC-ND** — no redistributing modified data. Prefer FieldPlant (CC BY 4.0) if you must redistribute. |
| **Few-shot / novel-class** | **PlantWild v2** (115 classes, long-tail, +text) | Many classes + long tail + text → ideal for episodic few-shot and CLIP/multimodal meta-learning across new crops. | Hold out several classes **entirely** as *novel*; seen only at few-shot eval (see [§16](#16-few-shot--meta-learning), [§26](#26-few-shot-evaluation-protocol)). |

**If the end goal is a deployable *single-crop field* classifier** (not a benchmark baseline), swap the **primary** to a field set: **Cassava-2020** (21,367 field imgs, 5 classes, 6.19 GB) or **Paddy Doctor** (16,225 field imgs, 13 classes, +variety/age metadata). Keep PlantDoc/FieldPlant as the cross-dataset test.

**Masks & text (modality sources):** if segmentation-aware training is pursued, pull masks from **PlantSeg** (11,400 masked + 8,000 healthy) or PlantVillage's segmented set. **Symptom text** exists only in **PlantWild**; for all other datasets, symptom text is **SYNTHETIC/DERIVED** and must be leakage-controlled ([§6](#6-dataset-preparation)). **Weather** exists in *no* dataset → it is joined from an external API ([§11](#11-weather-pipeline)) and is the highest leakage risk.

**Assumption flagged:** the multimodal fusion story assumes we *construct* image+text+weather examples from these unimodal sources. That construction — and its leakage controls — is itself a contribution ([§29.4](#294-positioning--how-this-project-differs-honest-framing-not-a-novelty-claim)), and a risk to be validated, not assumed.

---

## 6. Dataset Preparation

### 6.1 Data provenance labels (must appear in your report and defense)

| Modality | Type | How obtained | Ground-truth? |
|---|---|---|---|
| Leaf image | **REAL** | Public datasets | ✅ yes |
| Disease label | **REAL** | Dataset annotations | ✅ yes |
| Segmentation mask | **REAL** or **DERIVED** | dataset masks if present, else `rembg`/SAM output | ⚠️ derived masks are approximate |
| Symptom text | **DERIVED / SYNTHETIC** | templated + paraphrased from *symptom knowledge base per disease* | ❌ not human field text |
| Voice audio | **SYNTHETIC** | TTS of the symptom text (and/or a few real recordings) | ❌ synthetic |
| Weather | **REAL-but-attached** or **SYNTHETIC** | real historical weather by lat/lon/date *if the image has geo/time metadata*; otherwise simulated and labeled as demo-only | ⚠️ depends on metadata |

> **State this explicitly in the thesis.** Reviewers respect honest provenance far more than a vague "multimodal dataset". Synthetic augmentation is scientifically acceptable **as a stated method** with **leakage controls** and an **ablation** proving it isn't just re-encoding the label.

### 6.2 The leakage problem (the make-or-break of this project)

If symptom text is generated as `"{disease} symptoms: {canonical description}"`, then a text model trivially recovers the label and your "multimodal gain" is fake. **Controls:**
1. **Symptom-only vocabulary.** Templates describe *observable symptoms* (color, shape, location, texture) — never the disease name. e.g., Rust → *"small orange-brown pustules on the underside, yellowing around spots."*
2. **Many-to-one + one-to-many.** Multiple diseases share symptom phrases (spots, yellowing); one disease has many phrasings. This makes text *informative but not deterministic*.
3. **Paraphrase + noise.** Generate N paraphrases per sample; drop/reorder symptoms; add ASR-like noise for the voice path.
4. **Disjoint templates across splits.** Train templates ≠ val/test templates, so the text encoder can't memorize surface forms.
5. **Report text-only accuracy.** If a text-only classifier already hits ~100%, your text is leaking — fix the templates. Text-only should be *moderate*, meaningfully below image+text.
6. **Weather leakage:** never sample weather conditioned on the label. Attach real weather by location/date, or sample from a label-independent climatology, then ablate.

### 6.3 Grouping, dedup, balancing (see also [§17](#17-experiment-plan))
- **Grouped split:** if multiple images come from the same physical plant/leaf/source photo, keep the whole group on one side of train/val/test (`GroupShuffleSplit` on a `group_id`). Prevents "same leaf in train and test".
- **Dedup:** perceptual hash (`imagehash.phash`) to drop near-duplicates *before* splitting.
- **Stratify** by class so rare diseases appear in every split (except deliberately held-out novel classes).
- **Imbalance:** class-weighted loss or a `WeightedRandomSampler`; report **macro-F1**, not just accuracy.

### 6.4 Final `metadata.csv` (single source of truth)

```csv
sample_id,image_path,mask_path,disease,plant,split,group_id,
symptoms_text,text_source,voice_path,voice_source,
temperature,humidity,rainfall,wind_speed,latitude,longitude,weather_source,
image_source_dataset,is_novel_class
```

| Column | Provenance | Notes |
|---|---|---|
| `image_path`,`disease`,`plant`,`image_source_dataset` | **REAL** | from dataset |
| `mask_path` | **REAL/DERIVED** | dataset or rembg/SAM |
| `split`,`group_id`,`is_novel_class` | **DERIVED** | our split logic |
| `symptoms_text`,`text_source` | **DERIVED/SYNTHETIC** | `text_source ∈ {template, paraphrase, human}` |
| `voice_path`,`voice_source` | **SYNTHETIC/REAL** | `voice_source ∈ {tts, human}` |
| `temperature…wind_speed`,`weather_source` | **REAL-attached/SYNTHETIC** | `weather_source ∈ {api_historical, api_current, simulated}` |

Every derived/synthetic column carries a `*_source` flag so any experiment can filter to *real-only* and re-check results.

---

## 7. Image Pipeline

```
Raw image
  → validate (is-image? decode? min-resolution? not-corrupt?)
  → EXIF-orient + convert RGB
  → (optional) denoise / CLAHE for field images
  → leaf detect + background removal  →  mask
  → crop to leaf bbox (largest component)
  → resize 224×224 (aspect-preserve + pad, or RandomResizedCrop in train)
  → normalize (ImageNet mean/std)
  → augment (train only)
  → tensor [3,224,224]  →  ViT
```

### 7.1 Segmentation / background-removal options

| Approach | Cost | Needs masks? | MVP? | Notes |
|---|---|---|---|---|
| **Classical** (HSV/ExG green mask + largest contour) | tiny, CPU | no | ✅ fallback | Fails on yellow/necrotic leaves & green backgrounds. |
| **`rembg` (U²-Net)** | small, CPU/GPU | no (pretrained) | ✅ **MVP pick** | One-line clean cutout; robust enough; deterministic. |
| **U-Net / DeepLabv3** | medium | **yes** (train) | ⚠️ | Only if a leaf-mask dataset is available; more work. |
| **SAM / SAM2** | large (GPU) | no (promptable) | ➡️ research | Best masks; heavy; great for the "advanced" ablation. |

**Decision:** **MVP = `rembg`** (with classical fallback if `rembg` unavailable); **Research-grade = SAM2** with a center/foreground point prompt. Report *accuracy with segmentation ON vs OFF* — this ablation is one of your most convincing robustness results.

### 7.2 Fixed choices
- **Input size:** 224×224 (matches ViT-B/16). (384 optional for research at higher cost.)
- **Normalization:** ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]` (must match the pretrained ViT).
- **Train augmentations (disease-safe):** horizontal/vertical flip, ±20° rotation, mild scale/translation, **mild** brightness/contrast. **Avoid heavy hue shifts** — color is a disease cue; don't destroy it. Optional: RandomResizedCrop(0.7–1.0), CoarseDropout (small).
- **Val/Test:** resize + center-crop + normalize only. No TTA in the honest baseline.
- **Splits:** grouped + stratified, e.g. **70/15/15**; novel classes excluded from train entirely.
- **Imbalance:** `WeightedRandomSampler` or class-weighted loss; verify per-class counts.
- **Dedup:** phash before splitting.
- **Leakage:** background removal reduces background-source leakage; grouped split removes same-plant leakage; cross-dataset test removes dataset-source leakage.

---

## 8. ViT Pipeline

**Recommended model:** `google/vit-base-patch16-224` (ImageNet-1k fine-tuned) or `...-in21k` (ImageNet-21k pretrained, better for transfer). License/size verified in [§3](#3-recommended-technology-stack) model table (Apache-2.0, 86.6M).
**Alternatives to benchmark:** `facebook/dinov2-base` (strong frozen features), a `timm` ViT, and an EfficientNet/ResNet CNN baseline.

**Tensor contract:**
```
input        x        : [B, 3, 224, 224]
patch embed  →         : 196 patches (14×14) + 1 CLS = 197 tokens
hidden states          : [B, 197, 768]         # 768 = ViT-B hidden size (D_image)
pooled  z_image        : [B, 768]              # CLS token  (or mean-pool of patch tokens)
projection head        : Linear(768 → 256) + LN → z_img_proj [B, 256]
```
- **D_image = 768** (ViT-B/16). For cross-attention we keep the **full patch-token sequence `[B,197,768]`** (see [§12](#12-cross-attention-architecture)); for the concat MVP we use the pooled `[B,768]→[B,256]`.
- **Freeze vs fine-tune:**
  - **MVP:** freeze the entire ViT; train only the projection head + classifier/prototype space. Fast, CPU-trainable, low overfitting risk.
  - **Research-grade:** unfreeze the **last 2–4 transformer blocks + LayerNorm** with a small LR (e.g. 1e-5 backbone / 1e-3 head, discriminative LRs), keep early blocks frozen.
- **Head choice:** for the prototype design, the ViT is a **feature extractor**; the "classifier" is cosine similarity to prototypes. Optionally keep a parallel linear softmax head for a baseline and for the CE term in the combined loss ([§16](#16-loss-functions)).

---

## 9. Voice + Whisper Pipeline

```
audio (wav/webm) → resample 16kHz mono → (trim silence) → Whisper → raw text
 → clean (lowercase, strip fillers, normalize units) → symptom text → text encoder → z_text
```

- **Recommended model:** **`whisper small`** as the quality/speed sweet spot; **`base`/`tiny`** for pure-CPU/low-latency; **`large-v3`** only for research/accuracy. Use **`faster-whisper`** (CTranslate2) for 2–4× speed and lower RAM on CPU. Exact HF/repo IDs, sizes, languages, license verified in [§3](#3-recommended-technology-stack) model table (244M, permissive — MIT on OpenAI repo / Apache-2.0 on HF mirrors).
- **Why small:** multilingual, runs offline on a laptop, ~few-hundred-MB, good WER for short clear utterances.
- **Audio contract:** mono, **16 kHz**, ≤ **30 s** per clip (Whisper's window); accept `webm/opus` from the browser and transcode with ffmpeg.
- **Offline vs API:** run **offline/local** (privacy + no per-call cost). No cloud ASR needed.
- **Honesty note:** the model consumes **text**, not audio features — Whisper is a front-end. In experiments, "voice" and "typed text" are the *same* input to the model; measure ASR quality (WER) separately from disease accuracy.
- **De-risking:** implement the **typed-text path first**; add Whisper as a drop-in that produces the same `symptoms_text`.

---

## 10. Text Encoder Pipeline

```
symptoms_text → tokenizer → [input_ids, attention_mask] → encoder
   → token embeddings [B, L, H]  → pool (CLS or mean) → z_text [B, H] → proj [B, 256]
```

**Candidates to benchmark (don't assume BioBERT wins):** exact IDs/licenses verified in [§3](#3-recommended-technology-stack).

| Model | Emb dim H | Pool | Why consider |
|---|---|---|---|
| `all-MiniLM-L6-v2` (sentence-transformers) | **384** | mean (built-in) | **MVP pick**: tiny, fast, strong sentence semantics. |
| `bert-base-uncased` | 768 | CLS/mean | General baseline. |
| BioBERT (`dmis-lab/...`) | 768 | CLS/mean | *Biomedical* domain — unproven for botany; test it. |
| PubMedBERT/BiomedBERT (`microsoft/...`) | 768 | CLS/mean | Same caveat; strong on technical terms. |

- **Max sequence length:** symptom text is short → **64–128 tokens** (cap; model max is 512). Shorter = faster.
- **Pooling:** MiniLM has mean-pooling built in; for BERT use mean-pool over `attention_mask` (more stable than CLS for similarity).
- **Projection:** `Linear(H → 256)` so text lives in the same 256-d fused width as image/weather.
- **For cross-attention:** keep the **token sequence `[B, L, 768]`** (BERT) as keys/values, not just the pooled vector.
- **Fine-tuning:** MVP freezes the encoder; research-grade optionally fine-tunes on the symptom corpus (watch leakage — [§6.2](#62-the-leakage-problem-the-make-or-break-of-this-project)).

`z_text` contract: **z_text = [B, 256]** after projection (pooled path) / **[B, L, 256]** (token path for cross-attention).

---

## 11. Weather Pipeline

**API choice (VERIFIED): Open-Meteo** (`open-meteo.com`) — free, **no API key, no credit card**, and the only option supporting **both current/forecast AND a historical archive (ERA5, back to 1940)** by lat/lon with the exact variables we need. Alternatives were rejected for the free MVP: OpenWeather One Call 3.0 needs a card-backed subscription even for its free 1,000/day; WeatherAPI.com and Weatherbit gate historical data behind paid tiers.

| API | Key? | Free tier | Current | Historical | Verdict |
|---|---|---|---|---|---|
| **Open-Meteo** | **none** | free (non-commercial) | yes (forecast API) | **archive to 1940 (ERA5)** | ✅ **MVP pick** |
| NASA POWER | none | free | near-real-time (multi-day lag) | daily to 1981 | ✅ complement (deep agro-climate history) |
| OpenWeather One Call 3.0 | key + card | 1,000/day | yes | timemachine to 1979 | ⚠️ card required |
| WeatherAPI.com | key | 100k/month | yes | free = 1 day; since 2010 paid | ⚠️ history paid |
| Weatherbit.io | key | 50/day | yes | not on free (5y paid) | ❌ history gated |

**Endpoints (Open-Meteo):**
```
# historical (attach real weather for an image's location + capture date — no leakage)
GET https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}
    &start_date={YYYY-MM-DD}&end_date={YYYY-MM-DD}
    &hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m
# current (live diagnosis context)
GET https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}
    &current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m
```
**Variable mapping:** `temperature_2m`→temperature, `relative_humidity_2m`→humidity, `precipitation`→rainfall, `wind_speed_10m`→wind_speed. Optional extras: `surface_pressure`, `cloud_cover`, `soil_moisture_0_to_7cm`, `shortwave_radiation`, `et0_fao_evapotranspiration` (leaf-wetness/degree-day proxies).

**Raw variables (N_weather ≈ 6–10):**
```
temperature (°C), relative_humidity (%), rainfall/precip (mm),
wind_speed (m/s), [optional: pressure, cloud_cover, soil_moisture,
leaf_wetness_proxy, growing_degree_days]
```

**Normalization → embedding:**
```
raw vector       : [B, N_weather]          # e.g. N_weather = 8
standardize      : (x - μ_train) / σ_train  # μ,σ computed on TRAIN only (no leakage)
missing handling : impute 0 + concat binary missing-mask  → [B, 2*N_weather]
MLP encoder      : Linear(2N → 64) → LayerNorm → ReLU → Dropout(0.1)
                   → Linear(64 → 64) → ReLU → Linear(64 → 256)
z_weather        : [B, 256]                 # projected to common width
```
- **μ/σ from training split only**, saved with the model (applied identically at inference).
- **Missing weather:** if the API fails or no location, pass zeros + mask; the MLP learns a "no-weather" behavior. This is the [§32]/[§30](#30-risks-and-limitations) missing-modality contract.
- **Skeptic's checklist (put in the report):** weather branch must be **ablated** (with/without). If it adds ~0, say so — that's a legitimate, honest finding, and likely the truth for single-leaf lab data.

---

## 12. Cross-Attention Architecture

**Common fusion width:** `d_model = 256`. **Heads:** `h = 8` (head dim 32). **Layers:** `N = 2` (MVP-research) to `4`. **FFN dim:** `1024`. All blocks: residual + LayerNorm + GELU + dropout 0.1.

### Approach A — Concatenation + MLP (MVP)
```
z_img  [B,256]  ─┐
z_text [B,256]  ─┼─ concat → [B,768] → LN → Linear(768→256) → GELU → Dropout
z_wx   [B,256]  ─┘                     → Linear(256→256) → fused z_f [B,256]
```
Honest note: with only three pooled vectors this is a strong, hard-to-beat baseline. Report it.

### Approach B — Multimodal Transformer (recommended research-grade)
The key design decision: **attend over token sequences, not pooled vectors.** Use ViT **patch tokens** and BERT **token embeddings**; weather is one token. Self-attention over the concatenated multimodal sequence *is* cross-modal attention (every image patch can attend to every symptom word and to weather).

```
Project to d_model=256:
  image tokens   Wi: [B,197,768] → [B,197,256]
  text tokens    Wt: [B,L,768]   → [B,L,256]      (L ≤ 128)
  weather token  Ww: [B,256]     → [B,1,256]
  learned FUSION token:            [B,1,256]

Build sequence (+ modality-type embedding ∈ {fusion,img,txt,wx}, + keep ViT/BERT positions):
  H0 = [FUSION ; img(197) ; txt(L) ; wx(1)]   → [B, 199+L, 256]

Transformer encoder ×N  (MHA h=8, FFN 1024, residual+LN):
  for each layer:
     A  = MHA(Q=K=V=H)               # Q,K,V ∈ [B,199+L,256]; per-head 32-d
     H  = LN(H + A)
     H  = LN(H + FFN(H))
  HN : [B, 199+L, 256]

Fused embedding:
  z_f = HN[:,0,:]                     # the FUSION token → [B,256]
```

**Explicit directional cross-attention variant (MulT / ViLBERT style)** — if a reviewer wants literal "image→text / text→image":
```
img' = LN(img + CrossAttn(Q=img, K=txt, V=txt))     # image attends to text
txt' = LN(txt + CrossAttn(Q=txt, K=img, V=img))     # text attends to image
# weather via FiLM: γ,β = MLP(z_wx); img' = γ⊙img' + β  (and/or on txt')
z_f  = [meanpool(img') ; meanpool(txt')] → Linear → [B,256]
```

**Recommendation:** ship **A** first (MVP), then **B (full multimodal transformer)** as the headline research module because it is simpler to implement correctly than hand-wired directional co-attention, and it subsumes it. Keep the directional variant as an ablation. **Q/K/V:** in B, all three come from the joint sequence (self-attention); in the directional variant, Q is the "receiving" modality, K/V the "sending" one. Always **mask** missing-modality tokens in attention.

**Why this earns its place (and how to prove it):** compare A vs B vs image-only. If B doesn't beat A beyond noise, say so — but the *token-level* design is what gives B a real chance, and it gives you interpretable cross-modal attention maps (symptom word → lesion patch) for the thesis.

---

## 13. Prototype Memory

**Definition.** For class *k* with support embeddings `f(x_i)` (fused, L2-normalized):
```
        1
c_k =  ───  Σ   f(x_i)           then   ĉ_k = c_k / ‖c_k‖      (store normalized)
       |S_k| i∈S_k
```
**Prediction (cosine similarity, matches the diagram):**
```
sim_k = cos(f(q), ĉ_k) = ⟨ f̂(q), ĉ_k ⟩
p(k|q) = softmax_k( sim_k / τ )          # τ ≈ 0.05–0.1 (temperature)
ŷ = argmax_k sim_k ,   confidence = max_k p(k|q)
```
**Open-set / uncertainty:** if `max_k sim_k < θ_reject` → return *"uncertain — consult expert"* instead of a class ([§30](#30-risks-and-limitations)/[§32]). Tune `θ_reject` on validation (e.g., 5th percentile of correct-match similarities).

| Aspect | MVP | Research-grade |
|---|---|---|
| Prototype creation | mean of normalized train embeddings per class | class-balanced episodic means |
| # per class | 1 | K (k-means sub-prototypes) for multi-modal classes |
| Update | recompute on demand | EMA: `ĉ_k ← normalize(m·ĉ_k + (1−m)·batch_mean)` |
| Storage | `prototypes.pt` = `[num_classes, 256]` + `labels.json` | FAISS / pgvector if #prototypes large |
| Similarity | cosine | cosine (+ learned Relation-Net metric, optional) |
| Confidence | softmax over cosine/τ | + temperature calibration ([§18](#18-evaluation-metrics)) |

**Persistence:** a `[C,256]` float32 tensor is ~`C×1KB` — trivially a `.pt`/`.npy` file loaded at API startup. A vector DB is unnecessary until thousands of prototypes; note it as a scale path, don't build it for the MVP.

**Adding a new disease (no backbone retraining):** collect K images → embed with the frozen fused encoder → mean+normalize → append row to `prototypes.pt` and label to `labels.json`. Done. This is the concrete mechanism behind the diagram's "adapt to new diseases".

---

## 14. Meta-Learning / Few-Shot Architecture

### Algorithm choice
| Method | 2nd-order? | Impl effort | Fit here | Verdict |
|---|---|---|---|---|
| **Prototypical Networks** | no | low | excellent | ✅ **primary** |
| Matching Networks | no | medium | good | compare |
| Relation Network | no | medium | good (learned metric) | compare (research) |
| MAML | yes | high | overkill | mention only |
| Reptile | no (1st-order) | medium | ok | mention/compare |

**Decision:** **Prototypical Networks** — no second-order gradients, cheap, stable, and it *is* exactly the prototype+cosine design already in the architecture. This makes the "meta-learning" block concrete and defensible rather than buzzwordy.

### Episodic training (N-way K-shot)
```
Repeat for many episodes:
  1. Sample N classes from the meta-train class pool           (e.g., N=5)
  2. For each class sample K support + Q query images          (e.g., K=5, Q=15)
  3. Embed all with f_φ (ViT→fusion encoder)
  4. Prototypes: c_k = mean of the K support embeddings (normalize)
  5. For each query q: p(k|q)=softmax(-d(f(q),c_k)) or softmax(cos/τ)
  6. Loss = mean over queries of  −log p(y_q | q)               (episodic NLL)
  7. Backprop → update f_φ (and projection/fusion; ViT usually frozen or last blocks)
```

### Equations
```
Prototype:            c_k = (1/K) Σ_{i∈S_k} f_φ(x_i)
Squared-Euclidean:    d(q,c_k) = ‖f_φ(q) − c_k‖²          (original ProtoNet)
Cosine (our choice):  s(q,c_k) = ⟨ f̂_φ(q), ĉ_k ⟩
Class probability:    p(y=k|q) = exp(s(q,c_k)/τ) / Σ_{k'} exp(s(q,c_{k'})/τ)
Episode loss:         J = −(1/|Q|) Σ_{q∈Q} log p(y=k*_q | q)
```

### How new diseases are added / evaluated
- **Novel-class protocol:** keep a set of disease classes **entirely out of meta-training**. At test, give K support examples of each novel class → build prototypes → classify novel queries. Report **1/5/10-shot** accuracy and **known-class vs novel-class** separately. This is the *only* honest way to back the "generalizes to new diseases" claim ([§17](#17-experiment-plan)).
- **Confidence intervals:** average over ≥600 sampled episodes; report mean ± 95% CI (few-shot numbers are noisy).

---

## 15. Training Strategy

**Frozen (pretrained, not trained):** ViT backbone (MVP), Whisper (always), text encoder (MVP). **Trainable (small):** projection heads, weather MLP, fusion module, (research) last ViT blocks + optional text fine-tune.

| Stage | Goal | Train | Freeze | Skip for MVP? |
|---|---|---|---|---|
| **1** | Image baseline: ViT features → classifier/prototypes; establish honest accuracy + **cross-dataset** number | proj head (+ optional last ViT blocks) | ViT (mostly), all else | ❌ do first |
| **2** | Add text branch; align image+text (leakage-controlled); optional image–text contrastive | text proj (+ image proj) | encoders | ⚠️ MVP: pooled concat ok |
| **3** | Weather MLP on standardized features | weather MLP | rest | ✅ can skip early |
| **4** | Fusion (A then B) end-to-end on the multimodal set | fusion + proj heads | ViT/BERT/Whisper | ⚠️ MVP: Approach A |
| **5** | Prototype / metric space via episodic training (ProtoNet) + combined loss | fusion + proj (+ last ViT blocks) | pretrained bulk | ➡️ research core |
| **6** | Light end-to-end fine-tune (small LR) + calibration | last blocks + fusion | early blocks | ➡️ last, optional |

**MVP path = Stages 1 → (2 pooled) → 4A → prototype classifier.** **Research path adds 3, 4B, 5, 6 + ablations.** Always keep Stage-1's cross-dataset number as the anchor everything is compared against.

**Optimization defaults (starting points, not gospel):** AdamW, head LR 1e-3, backbone LR 1e-5, cosine schedule + warmup, weight decay 0.05, batch 32 (classification) / episodic batch for ProtoNet, early-stop on val macro-F1, mixed precision (`torch.cuda.amp`) on GPU.

---

## 16. Loss Functions

| Loss | Role | Use when |
|---|---|---|
| **Cross-entropy** (softmax head) | stable baseline, Stage 1 | always available as anchor |
| **Prototypical (episodic NLL)** | shapes prototype space, few-shot | Stage 5 (primary metric loss) |
| **Supervised contrastive** | tight same-class clusters | Stage 5 pretrain of embedding (optional) |
| Triplet / contrastive | classic metric | if SupCon unstable / small batch |

**Combined objective (research-grade):**
```
L_total = λ1·L_CE  +  λ2·L_metric(SupCon or triplet)  +  λ3·L_proto
```
**Initial λ (starting point — tune on val, do NOT present as optimal):** `λ1=1.0, λ2=0.5, λ3=1.0`. Rationale: CE stabilizes early training; prototypical loss is the task-aligned objective (equal weight); metric loss is a regularizer (down-weighted). **Tune via a small sweep** and report sensitivity. For **MVP**, `L = L_CE` alone is fine; introduce `L_proto` at Stage 5.

> Explicitly state in the report: these weights are dataset-dependent hyperparameters chosen by validation, not universal constants.

---

## 17. Experiment Plan

### Baselines / model ladder (each is a claimable result)
| # | Model | Demonstrates |
|---|---|---|
| B1 | Image-only CNN (EfficientNet/ResNet) | non-transformer floor |
| B2 | Image-only ViT | transformer image floor + prototype-vs-softmax |
| B3 | Image + text (concat) | does text help (leakage-controlled)? |
| B4 | Image + weather (concat) | does weather help (likely ~0 — report honestly)? |
| B5 | Image + text + weather (concat, Approach A) | full multimodal, simple fusion |
| B6 | Multimodal + **cross-attention** (Approach B) | does token-level fusion beat concat? |
| B7 | B6 + **prototype/metric** training | does metric space help? |
| B8 | B7 trained **episodically** (ProtoNet) | few-shot + novel-class ability |

### Ablation table (fill with your runs)
| Exp | Components | Acc | Macro-F1 | Recall | Precision | AUROC |
|---|---|---|---|---|---|---|
| B2 | img(ViT) | | | | | |
| B5 | img+txt+wx (concat) | | | | | |
| B6 | + cross-attn | | | | | |
| B7 | + metric | | | | | |
| B8 | + episodic | | | | | |
| −seg | B6 without segmentation | | | | | |
| −txt | B6 without text | | | | | |
| −wx | B6 without weather | | | | | |

### Few-shot study
- **1-shot / 5-shot / 10-shot**, 5-way (and N-way if classes allow), ≥600 episodes, mean ± 95% CI.
- **Known vs novel classes** reported separately.

### Cross-dataset generalization (the credibility centerpiece)
- **Train on Dataset A (lab) → test on Dataset B (field).** Report the accuracy drop. Then show few-shot adaptation on B recovers part of it. *This is the experiment that makes the whole "robust, adaptive" claim real.*

---

## 18. Evaluation Metrics

Implement all; report the right ones per experiment.
- **Accuracy** — headline, but *misleading under class imbalance* (a model that always predicts the majority disease can look "good"). Always pair with macro metrics.
- **Precision / Recall / F1** — per class + **macro** (unweighted mean → treats rare diseases equally) and **weighted**.
- **Macro-F1** — primary model-selection metric here.
- **Confusion matrix** — reveals which diseases are confused (e.g., early blight vs late blight); include in the report.
- **ROC-AUC / PR-AUC** — one-vs-rest, macro-averaged; PR-AUC is more informative under imbalance.
- **Top-k accuracy** (k=3) — useful for a "possible diseases" UI.
- **Calibration** — **ECE** + reliability diagram; temperature-scale the similarity softmax so displayed confidence is trustworthy (a "92%" must mean ~92%).
- **Few-shot accuracy** — mean ± 95% CI over episodes; separate known/novel.
- **Segmentation** (if evaluated) — IoU/Dice on any available masks.

Report **both** in-distribution and **cross-dataset** for every headline model.

---

## 19. Backend Architecture

```
React frontend
   │  HTTPS (JSON + multipart)
   ▼
FastAPI (Uvicorn)  ── auth (JWT) ── rate-limit ── validation (pydantic)
   │
   ├─ InferenceService (singleton, models loaded ONCE at startup)
   │     ├─ ImagePreprocessor (rembg + resize + normalize)
   │     ├─ ViTEncoder            → z_image
   │     ├─ WhisperService (ASR)  → text        [POST /transcribe]
   │     ├─ TextEncoder           → z_text
   │     ├─ WeatherService (API)  → weather vec  [GET /weather]
   │     ├─ WeatherEncoder (MLP)  → z_weather
   │     ├─ FusionModel           → z_fused
   │     └─ PrototypeClassifier   → (disease, confidence, top-k)
   │
   ├─ RecommendationService (curated KB lookup — NOT an LLM)
   └─ Repository (SQLAlchemy) → PostgreSQL/SQLite
```

- **Model loading:** load ViT/Whisper/text encoder/fusion/prototypes **once** into a module-level singleton on app startup (`lifespan`), not per request. Inference is CPU-OK for ViT+prototype; Whisper `small` benefits from a few CPU threads or GPU.
- **Sync vs async:** MVP = synchronous request → response (a diagnosis is ~0.5–3 s on CPU). If Whisper large / SAM make it slow, move heavy jobs to **Celery + Redis** and return a job id (research/scale).
- **Separation of concerns:** ML diagnosis and agricultural advice are different services; the KB is versioned data, reviewed by a human.

---

## 20. Database Schema

Persist **decisions and metadata, not big tensors.** Store prototypes as files/artifacts; store only a *reference/version* in the DB.

```sql
CREATE TABLE users (
  id            SERIAL PRIMARY KEY,
  email         VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at    TIMESTAMP DEFAULT now()
);

CREATE TABLE diseases (               -- reference/KB table
  id            SERIAL PRIMARY KEY,
  key           VARCHAR(64) UNIQUE NOT NULL,   -- 'rust','late_blight',...
  display_name  VARCHAR(128) NOT NULL,
  plant         VARCHAR(128),
  description    TEXT,
  typical_symptoms TEXT
);

CREATE TABLE recommendations (        -- curated, human-reviewed
  id            SERIAL PRIMARY KEY,
  disease_id    INT REFERENCES diseases(id),
  severity      VARCHAR(32),
  action_steps  JSONB,               -- ["Remove infected leaves", ...]
  prevention    JSONB,
  source        VARCHAR(255),        -- provenance of the advice
  reviewed_by   VARCHAR(128)
);

CREATE TABLE diagnoses (
  id            SERIAL PRIMARY KEY,
  user_id       INT REFERENCES users(id),
  image_path    VARCHAR(512) NOT NULL,       -- object storage key
  audio_path    VARCHAR(512),
  symptoms_text TEXT,
  predicted_disease_id INT REFERENCES diseases(id),
  confidence    REAL,
  topk          JSONB,               -- [{"disease":"rust","p":0.92}, ...]
  model_version VARCHAR(64),
  prototype_version VARCHAR(64),
  created_at    TIMESTAMP DEFAULT now()
);

CREATE TABLE weather_data (          -- attached context per diagnosis
  id            SERIAL PRIMARY KEY,
  diagnosis_id  INT REFERENCES diagnoses(id),
  latitude      REAL, longitude REAL,
  temperature   REAL, humidity REAL, rainfall REAL, wind_speed REAL,
  source        VARCHAR(64),         -- 'open-meteo-historical' | 'simulated'
  observed_at   TIMESTAMP
);

-- OPTIONAL, only if you scale prototypes into the DB (else keep prototypes.pt on disk)
CREATE TABLE prototype_metadata (
  id SERIAL PRIMARY KEY, disease_id INT REFERENCES diseases(id),
  version VARCHAR(64), num_support INT, created_at TIMESTAMP DEFAULT now()
);
```
**Do NOT** store per-image 256-d embeddings row-by-row unless you have a reason (e.g., online prototype recompute). If you do, use `pgvector`. For the MVP, prototypes live in `prototypes.pt`.

---

## 21. Frontend Architecture

| Screen | Purpose | Key data flow |
|---|---|---|
| 1. Login/Register | auth | `POST /auth/*` → JWT stored |
| 2. Capture/Upload leaf | image input | camera/file → preview → hold blob |
| 3. Record voice (optional) | symptoms | `MediaRecorder` → webm blob; or typed text |
| 4. Processing | feedback | multipart `POST /diagnose`; spinner/progress |
| 5. Diagnosis result | disease + confidence + top-k bar | render response JSON |
| 6. Disease details | symptoms, severity, recommendations, prevention, weather context | from same response / `GET /diseases/:key` |
| 7. History | past diagnoses | `GET /history/:userId` |

```
[capture image] + [record voice? → blob] + [geolocation? → lat/lon]
      │
      ▼  multipart/form-data
POST /api/diagnose  ──►  {disease, confidence, topk, symptoms, weather, recommendations}
      │
      ▼
render result (Screen 5) → details (Screen 6) → saved to history (Screen 7)
```
Handle **partial inputs** gracefully: image-only is valid; voice/weather are optional and the UI shows which modalities were used.

---

## 22. Project Folder Structure

```
manishSirMLLeafProject/
├── docs/                      # this plan, diagrams, thesis notes
├── configs/                   # yaml: data, model, train, api
├── data/
│   ├── raw/                   # downloaded datasets (gitignored)
│   ├── interim/               # segmented/cropped
│   ├── processed/             # final tensors + metadata.csv
│   └── prototypes/            # prototypes.pt, labels.json
├── ml/
│   ├── data/                  # dataset_loader.py, episodic_sampler.py, build_metadata.py
│   ├── preprocessing/         # image_preprocessor.py, segmentation.py, text_synth.py, weather_attach.py
│   ├── models/                # vit_encoder.py, text_encoder.py, weather_encoder.py,
│   │                          #   cross_attention.py, multimodal_model.py, prototype_memory.py, proto_net.py
│   ├── training/              # trainer.py, episodic_trainer.py, losses.py
│   ├── inference/             # pipeline.py, recommend.py
│   ├── evaluation/            # metrics.py, ablations.py, fewshot_eval.py, crossdataset_eval.py
│   └── services/              # whisper_service.py, weather_service.py
├── backend/
│   ├── app/                   # main.py (FastAPI), routers/, schemas/, deps/, auth/
│   ├── db/                    # models.py (SQLAlchemy), migrations/ (alembic)
│   └── kb/                    # diseases.json, recommendations.json (curated)
├── frontend/                  # React (Vite) app: src/screens, src/api, src/components
├── notebooks/                 # exploration, sanity checks, figures for thesis
├── scripts/                   # download_data.py, train.sh, export_prototypes.py
├── tests/                     # unit/ integration/ api/ e2e/
├── requirements.txt / pyproject.toml
├── docker-compose.yml
└── README.md
```
**Responsibilities:** `ml/` is training-time + reusable inference logic (importable by backend). `backend/` is the API + DB + curated KB. `frontend/` is UI only. `configs/` centralizes hyperparameters for reproducibility. `data/` is gitignored except `metadata.csv` schema + small samples.

---

## 23. API Design

All responses JSON; auth via `Authorization: Bearer <JWT>`. OpenAPI docs auto-served at `/docs`.

**`POST /api/diagnose`** — multipart: `image` (file, required), `audio` (file, optional), `text` (string, optional), `lat`,`lon` (optional).
```json
// 200 OK
{
  "diagnosis_id": 1421,
  "disease": "rust",
  "display_name": "Leaf Rust",
  "confidence": 0.92,
  "topk": [
    {"disease": "rust", "p": 0.92},
    {"disease": "blight", "p": 0.05},
    {"disease": "mildew", "p": 0.02}
  ],
  "symptoms_used": "yellow spots and brown edges",
  "modalities_used": {"image": true, "text": true, "weather": true},
  "weather": {"temperature": 27.4, "humidity": 82, "rainfall": 3.1, "wind_speed": 2.0,
              "source": "open-meteo-historical"},
  "severity": "moderate",
  "recommendations": ["Remove and destroy infected leaves",
                      "Apply an appropriate labelled fungicide per local guidance",
                      "Improve air circulation / avoid overhead irrigation"],
  "prevention": ["Use resistant varieties", "Rotate crops", "Monitor humidity"],
  "disclaimer": "AI-assisted suggestion; confirm with a local agronomist before chemical use.",
  "model_version": "vit-proto-v0.3", "low_confidence": false
}
```
```json
// 200 OK but below threshold
{ "disease": null, "confidence": 0.31, "low_confidence": true,
  "message": "Unable to diagnose reliably. Retake a clear, well-lit close-up of the affected leaf.",
  "topk": [ ... ] }
```

**`POST /api/transcribe`** — multipart `audio` → `{ "text": "there are yellow spots on the leaf", "language": "en", "duration_s": 6.2 }`

**`GET /api/weather?lat=..&lon=..&date=..`** → `{ "temperature":27.4, "humidity":82, "rainfall":3.1, "wind_speed":2.0, "source":"open-meteo-historical" }`

**`GET /api/history/{user_id}`** → `{ "items": [ {"diagnosis_id":1421,"disease":"rust","confidence":0.92,"created_at":"..."} ], "page":1 }`

**`POST /api/auth/register` · `POST /api/auth/login`** → `{ "access_token":"...", "token_type":"bearer" }`

Validation: reject non-image mime types, enforce max upload size, sanitize filenames, clamp lat/lon. Errors use consistent `{ "error": {"code":"...", "message":"..."} }`.

---

## 24. Exact Implementation Phases

Each phase: **Objective · Input → Output · Files · Libraries · Key commands · Validation · Common errors.** Dataset/model-specific download commands are consolidated in [START HERE](#start-here) and finalized once verified.

**Phase 0 — Environment**
- Obj: reproducible env. In: none → Out: working `venv`/conda + GPU/CPU check.
- Files: `requirements.txt`, `configs/*.yaml`. Libs: torch, transformers, timm, sentence-transformers, faster-whisper, rembg, albumentations, opencv-python, scikit-learn, pandas, fastapi, uvicorn, pytest.
- Cmd: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`; `python -c "import torch;print(torch.cuda.is_available(), torch.backends.mps.is_available())"`.
- Validate: imports succeed; device detected. Errors: CUDA/torch mismatch (install the torch build matching your CUDA/CPU/MPS).

**Phase 1 — Dataset download + cleaning**
- Obj: raw datasets on disk, deduped. In: dataset URLs → Out: `data/raw/*`, dedup report.
- Files: `scripts/download_data.py`, `ml/data/build_metadata.py`. Libs: kaggle, imagehash, pandas.
- Validate: image counts match the source; phash finds/records near-dups; class histogram printed. Errors: Kaggle API token missing (`~/.kaggle/kaggle.json`), corrupt downloads, mixed folder layouts.

**Phase 2 — Image preprocessing + segmentation**
- Obj: clean, background-removed, cropped, normalized images. In: `data/raw` → Out: `data/interim` + masks.
- Files: `ml/preprocessing/image_preprocessor.py`, `segmentation.py`. Libs: rembg, opencv, Pillow, albumentations.
- Validate: eyeball 30 random cutouts; measure IoU on any GT masks; assert no all-black masks. Errors: rembg model download offline; leaves cut off by aggressive masks (add fallback to classical/no-mask).

**Phase 3 — ViT image baseline** ← *do this before anything fancy*
- Obj: honest image-only accuracy + **cross-dataset** number. In: `data/interim` + `metadata.csv` → Out: `vit_baseline.pt`, metrics.
- Files: `ml/models/vit_encoder.py`, `ml/training/trainer.py`, `ml/evaluation/metrics.py`. Libs: transformers, sklearn.
- Validate: in-distribution macro-F1 sane (lab data often high); **report the drop on the field set** (expect a big drop — that's the finding). Errors: normalization mismatch with pretrained ViT; data leakage via ungrouped split (fix split first).

**Phase 4 — Voice → Whisper → text**
- Obj: audio to symptom text. In: wav/webm → Out: transcripts.
- Files: `ml/services/whisper_service.py`. Libs: faster-whisper, ffmpeg.
- Validate: WER on a few known clips; latency acceptable. Errors: ffmpeg not installed; wrong sample rate; huge model OOM (use `small`/`base`).

**Phase 5 — Text encoder**
- Obj: symptom text → `z_text`. In: text → Out: embeddings + a text-only classifier for the **leakage check**.
- Files: `ml/models/text_encoder.py`, `ml/preprocessing/text_synth.py`. Libs: sentence-transformers/transformers.
- Validate: **text-only accuracy is moderate, not ~100%** (else templates leak — fix §6.2). Errors: over-informative templates; tokenizer/length misconfig.

**Phase 6 — Weather API**
- Obj: fetch/attach weather by lat/lon/date. In: coords → Out: weather vectors + `weather_source` flags.
- Files: `ml/services/weather_service.py`, `ml/preprocessing/weather_attach.py`. Libs: requests/httpx.
- Validate: values in physical ranges; caching works; graceful failure → nulls+mask. Errors: rate limits, missing key, timezone/date handling.

**Phase 7 — Weather embedding (MLP)**
- Obj: `[B,N]→[B,256]`. In: standardized vectors → Out: `z_weather`.
- Files: `ml/models/weather_encoder.py`. Validate: μ/σ from **train only**; missing-mask path works. Errors: leaking test stats into normalization.

**Phase 8 — Cross-attention fusion**
- Obj: fuse modalities. In: z_image/z_text/z_weather (+ token seqs) → Out: `z_fused`, fusion model.
- Files: `ml/models/cross_attention.py`, `multimodal_model.py`. Validate: Approach A trains; then B ≥ A? report either way. Errors: shape/mask bugs; unstable attention (add LN/warmup); OOM from long token seqs (cap L, patch subsample).

**Phase 9 — Prototype memory**
- Obj: build/classify with prototypes. In: fused embeddings → Out: `prototypes.pt`, classifier.
- Files: `ml/models/prototype_memory.py`. Validate: cosine-to-prototype accuracy ≈ softmax head; add-new-class works. Errors: forgetting to L2-normalize; τ mis-set.

**Phase 10 — Few-shot / meta-learning (ProtoNet, episodic)**
- Obj: episodic training + novel-class eval. In: episodes → Out: meta-trained encoder, few-shot curves.
- Files: `ml/models/proto_net.py`, `ml/data/episodic_sampler.py`, `ml/training/episodic_trainer.py`, `ml/evaluation/fewshot_eval.py`.
- Validate: 1/5/10-shot with 95% CI; **novel-class classes truly unseen in meta-train**. Errors: class leakage across meta-splits; too-few classes for N-way (reduce N).

**Phase 11 — Inference API**
- Obj: one endpoint runs the whole pipeline. In: image(+audio+coords) → Out: JSON diagnosis.
- Files: `backend/app/main.py`, `ml/inference/pipeline.py`, `recommend.py`. Libs: fastapi, uvicorn.
- Validate: `POST /api/diagnose` returns valid schema; low-confidence path fires. Errors: models reloaded per request (use singleton); blocking I/O.

**Phase 12 — Frontend**
- Obj: capture → result UI. Files: `frontend/src/*`. Libs: react, vite, tailwind. Validate: camera+mic+geo permissions; partial-input flows. Errors: CORS, audio codec (webm/opus → transcode server-side).

**Phase 13 — Database + history**
- Obj: persist diagnoses/weather/users. Files: `backend/db/models.py`, alembic migrations. Validate: history endpoint paginates; no raw tensors stored. Errors: migration drift.

**Phase 14 — Testing** (see [§27](#27-testing)). **Phase 15 — Deployment** (see [§28](#28-deployment)).

---

## 25. MVP Version

**Goal: a complete, honest, demoable system in the shortest path.**
```
image → rembg → ViT(frozen) → z_image ─┐
typed/spoken text → (Whisper→) MiniLM → z_text ─┼─ concat+MLP (Approach A) → z_fused
weather (optional) → MLP → z_weather ──┘            │
                                                    ▼
                                   prototype memory (cosine) → disease + confidence
                                                    ▼
                                   curated KB → recommendations   →  React UI
```
**MVP simplifications (all deliberate):** frozen ViT + linear proj; MiniLM text (or typed text only first); concat fusion; single prototype/class; cross-entropy (or simple prototypical) — **no episodic meta-learning yet**; weather optional; `rembg` not SAM; SQLite; stub auth. **Still includes:** cross-dataset test, low-confidence rejection, honest ablation of text/weather. This is defensible on its own.

---

## 26. Research-Grade Version

Adds, each as a measured delta over the MVP:
1. **Segmentation ablation** (rembg vs SAM2 vs none).
2. **Token-level cross-attention** (Approach B) + attention-map visualizations.
3. **Metric space**: SupCon/prototypical + combined loss; embedding geometry analysis.
4. **Episodic meta-learning** (Prototypical Networks): 1/5/10-shot, **novel-class** protocol, ≥600 episodes ± CI; compare Matching/Relation/Reptile.
5. **Cross-dataset generalization**: train lab → test field, then few-shot adaptation recovery.
6. **Calibration**: temperature scaling + ECE/reliability diagrams.
7. **Text-encoder study** (MiniLM vs BERT vs BioBERT vs PubMedBERT) with leakage controls.
8. **Weather causality check** (real-attached vs simulated vs none).

**Why staged (state this in the defense):** every research component is evaluated as an *improvement over a real baseline*, so you can attribute gains to specific ideas — which is exactly what turns "I built a big architecture" into "I demonstrated that X helps by Y%." Building the full stack at once makes failures unattributable and the thesis unfalsifiable.

---

## 27. Testing

| Level | Tool | Example cases |
|---|---|---|
| **Unit** | pytest | preprocessing output shapes/normalization; prototype math (cosine, argmax); weather normalization uses train stats; text templates don't contain disease names (leakage guard as a *test*) |
| **Model** | pytest | ViT output `[B,768]`; fusion output `[B,256]`; prototype classifier vs softmax parity; few-shot eval reproducible with fixed seed |
| **Integration** | pytest | full `pipeline.py` on a sample image returns valid schema; missing audio/weather handled |
| **API** | httpx/pytest | `POST /diagnose` happy path; invalid mime → 400; oversize upload → 413; low-confidence → `low_confidence:true`; auth required |
| **Frontend** | Vitest/RTL | screens render; partial-input flows; error toasts |
| **E2E** | Playwright | capture → diagnose → result → history |

**Must-have edge cases (from your list):** valid leaf ✓ · non-leaf/invalid image → reject ✓ · noisy/blurry → low-confidence ✓ · missing voice → image-only path ✓ · missing weather → mask path ✓ · unknown disease → open-set reject ✓ · low confidence → "unable to diagnose" ✓ · weather API down → graceful null ✓ · wrong format → 400 ✓ · very large image → resize/413 ✓.

---

## 28. Deployment

```
Frontend (static)      → Netlify/Vercel/GitHub Pages  (or Nginx on the VPS)
Backend API + ML       → Docker container on a VPS (CPU is enough for ViT+proto+small Whisper)
DB                     → managed Postgres or a container volume
Model artifacts        → mounted volume / object storage (prototypes.pt, weights)
```
- **MVP deploy:** one `docker-compose.yml` (api + db + nginx). Inference on **CPU** is viable (ViT-B + prototype is light; Whisper `small` is the heaviest). Keep a `MODEL_DEVICE` env toggle.
- **When to add GPU:** only if using SAM2 / Whisper large / heavy fine-tuning in production. Otherwise rent a cloud GPU **for training only**, deploy CPU inference.
- **Serverless:** fine for the stateless weather/transcribe helpers; the model service is better as a long-lived container (cold-start + model size make serverless model-loading painful).
- **Config/secrets:** API keys in env/secret manager, never in code; CORS locked to the frontend origin; image/audio retention policy configurable ([§33 security]).

---

## 29. Research / Literature Review

> Citation policy: every paper below was **verified against its arXiv/DOI page** (methods, applied, and dataset sets all confirmed). Venues not printed on an arXiv page are labeled as canonical/high-confidence. **Do not cite anything still marked UNVERIFIED.**

### 29.1 Methods & backbones (VERIFIED — arXiv-confirmed)

| # | Paper | Year | Venue | Link | Relevance |
|---|---|---|---|---|---|
| 1 | **Prototypical Networks for Few-shot Learning** — Snell, Swersky, Zemel | 2017 | NeurIPS | arXiv:1703.05175 | **Primary few-shot method**: nearest class-prototype in a learned metric space. |
| 2 | **Matching Networks for One Shot Learning** — Vinyals et al. | 2016 | NeurIPS | arXiv:1606.04080 | Episodic support/query template; compared baseline. |
| 3 | **MAML** — Finn, Abbeel, Levine | 2017 | ICML | arXiv:1703.03400 | Gradient-based meta-learning; mentioned, not primary. |
| 4 | **On First-Order Meta-Learning Algorithms (Reptile)** — Nichol, Achiam, Schulman | 2018 | arXiv/OpenAI | arXiv:1803.02999 | Cheap first-order meta-learning; optional comparison. |
| 5 | **Learning to Compare: Relation Network** — Sung et al. | 2018 | CVPR | arXiv:1711.06025 | Learned similarity metric; research-grade comparison. |
| 6 | **Supervised Contrastive Learning** — Khosla et al. | 2020 | NeurIPS | arXiv:2004.11362 | Metric-space pretraining for tight class clusters. |
| 7 | **FaceNet (triplet loss)** — Schroff et al. | 2015 | CVPR (DOI 10.1109/CVPR.2015.7298682) | arXiv:1503.03832 | Classic metric objective; optional. |
| 8 | **ViT: An Image is Worth 16×16 Words** — Dosovitskiy et al. | 2020/2021 | ICLR | arXiv:2010.11929 | **Image backbone.** |
| 9 | **DINOv2** — Oquab et al. | 2023 | TMLR* | arXiv:2304.07193 | Strong frozen features for few-shot; alt image encoder. |
| 10 | **Whisper** — Radford et al. | 2022 | ICML 2023 | arXiv:2212.04356 | **ASR** for the voice branch. |
| 11 | **BERT** — Devlin et al. | 2018/2019 | NAACL | arXiv:1810.04805 | Text-branch encoder baseline. |
| 12 | **BioBERT** — Lee et al. | 2019/2020 | Bioinformatics (DOI 10.1093/bioinformatics/btz682) | arXiv:1901.08746 | Biomedical text encoder (unproven for botany — test). |
| 13 | **PubMedBERT/BiomedBERT (Domain-Specific Pretraining)** — Gu et al. | 2020/2021 | ACM HEALTH (DOI 10.1145/3458754) | arXiv:2007.15779 | In-domain biomedical encoder; alt text branch. |
| 14 | **CLIP** — Radford et al. | 2021 | ICML | arXiv:2103.00020 | Image–text alignment; zero-shot/label-as-text idea. |
| 15 | **Multimodal Transformer (MulT)** — Tsai et al. | 2019 | ACL | arXiv:1906.00295 | **Cross-attention fusion recipe** for unaligned modalities. |
| 16 | **ViLBERT** — Lu et al. | 2019 | NeurIPS | arXiv:1908.02265 | Two-stream co-attention fusion alternative. |
| 17 | **Segment Anything (SAM)** — Kirillov et al. | 2023 | ICCV | arXiv:2304.02643 | Research-grade leaf/lesion segmentation. |
| 18 | **SAM 2** — Ravi et al. | 2024 | ICLR 2025* | arXiv:2408.00714 | Upgraded segmentation front-end. |

`*` DINOv2→TMLR and SAM2→ICLR 2025 venues are high-confidence but not printed on the arXiv page; verify before final citation.

### 29.2 Applied plant-disease literature (VERIFIED — arXiv / Crossref / Europe PMC)

| # | Paper | Year | Venue | Link | Method / dataset / result | Limitation |
|---|---|---|---|---|---|---|
| A1 | **ViT for automated plant disease classification** — Borhani, Khoramdel, Najafi | 2022 | Scientific Reports | DOI 10.1038/s41598-022-15163-0 | Lightweight ViT vs CNN & CNN-ViT hybrids on public sets | Accuracy↔latency trade-off; no single headline number |
| A2 | **PlantXViT: explainable ViT-CNN** — Thakur et al. | 2022 | arXiv (journal UNVERIFIED — cite arXiv) | arXiv:2207.07919 | ~0.8M-param CNN+ViT + Grad-CAM/LIME; >93–98% on Apple/Maize/Rice | curated crop subsets; field robustness less tested |
| A3 | **Snap and Diagnose (PlantWild)** — Wei, Chen, Yu | 2024 | arXiv | arXiv:2408.14723 | CLIP cross-modal **image↔text retrieval**; introduces PlantWild (18k+ imgs, 89 cats) | retrieval not calibrated classification |
| A4 | **Benchmarking In-the-wild Multimodal Disease Recognition** — Wei, Chen, Huang, Yu | 2024 | arXiv | arXiv:2408.03120 | **image + per-class text**, multi-prototype, few-shot/training-free | modest absolute accuracy; hard benchmark |
| A5 | **Plant Disease Detection via MLLMs + CNNs** — Roumeliotis, Sapkota, Karkee et al. | 2025 | arXiv | arXiv:2504.20419 | GPT-4o (MLLM) vs/with ResNet-50; fine-tuned GPT-4o 98.12% (apple) | zero-shot MLLM weak; only 2 crops, lab images |
| A6 | **Few-Shot Learning for field plant disease** — Argüeso, Picón, Irusta et al. | 2020 | Comput. Electron. Agric. | DOI 10.1016/j.compag.2020.105542 | Inception-V3 + Siamese + **triplet**; PlantVillage 32 source + 6 target; ~94% target, ~90% less data | validated on lab-style PlantVillage |
| A7 | **Few-shot cotton leaf-spot via metric learning** — Liang | 2021 | Plant Methods | DOI 10.1186/s13007-021-00813-7 | spot-seg + twin CNN metric learning + KNN; +~7.7% over DenseNet | single self-built dataset, one crop |
| A8 | **ML disease forecasting: rice blast** — Kaundal, Kapoor, Raghava | 2006 | BMC Bioinformatics | DOI 10.1186/1471-2105-7-485 | **weather-only** SVM vs regression/BPNN/GRNN; SVM best | tabular; **not image-fused** |
| A9 | **Deep models for long-term disease prediction (wheat yellow rust, England)** — Yuan, Zhang, Bi, Yang | 2025 | arXiv | arXiv:2501.15677 | FC-NN & **LSTM over historical weather**; ~83.65% @6-month | preliminary; weather-only, **not image-fused** |
| A10 | **Crop yield & disease from soil + weather** — Ahmed, Das, Zubair | 2024 | arXiv / IEEE iCACCESS | arXiv:2403.19273 | soil→SARIMAX weather→SVC disease-risk→DT yield | no quantitative metrics; region-specific; **not image-fused** |

*Leads (surfaced, not fully vetted — verify before citing):* Uzhinskiy 2025 "Evaluation of Different Few-Shot Learning Methods in the Plant Disease Classification Domain," *Biology*, DOI 10.3390/biology14010099.

### 29.3 Dataset papers (VERIFIED)

| Dataset | Paper | Year | Link | Scale (per paper) | Nature |
|---|---|---|---|---|---|
| **PlantVillage** | Hughes & Salathé | 2015 | arXiv:1511.08060 | ~54,000+ images, healthy + diseased | **lab** (plain backgrounds) |
| **PlantDoc** | Singh, Jain, Jain, Kayal, Kumawat, Batra | 2019/2020 | arXiv:1911.10317 · DOI 10.1145/3371158.3371196 | 2,598 images, 13 species, up to 17 disease classes | **field / internet-sourced** (label noise) |
| **PlantWild** | Wei, Chen, Yu | 2024 | arXiv:2408.14723 | 18,000+ images, 89 categories | **in-the-wild**, paired text descriptions |

> These per-paper figures anchor the numbers in [§4](#4-dataset-research)/[§5](#5-selected-datasets); the datasets agent adds licenses, exact download methods, and storage sizes.

### 29.4 Positioning — how this project differs (honest framing, not a novelty claim)
**Literature-scan result (important):** the *verifiable* multimodal plant-disease work is almost entirely **image + text** (CLIP retrieval, VLM/MLLM — A3, A4, A5). We found **no verifiable paper that fuses leaf images directly with weather** for per-leaf diagnosis; the weather-aware works (A8, A9, A10) are **tabular/time-series** on meteorology, never image-fused. So **image+weather(+text) fusion for per-leaf diagnosis is a genuinely under-explored niche** — a legitimate contribution, but one with little prior art to lean on (which also means you must *prove* its value, not assume it). The *defensible* contributions to demonstrate experimentally are: **(a)** a leakage-controlled protocol for constructing a multimodal (image+symptom-text+weather) dataset from unimodal sources; **(b)** an ablation quantifying each modality's *true* marginal value (hypothesis: image ≫ text > weather≈0); **(c)** token-level cross-attention vs concat fusion, measured; **(d)** prototype/episodic few-shot for **novel-disease** addition (1/5/10-shot; cf. A6/A7 which stay in-domain); **(e)** **cross-dataset** lab→field generalization (PlantVillage→PlantDoc/PlantWild) with few-shot recovery. Novelty = the *combination + honest evaluation*, framed against A1–A10.

---

## 30. Risks, Limitations & Computational Requirements

### 30.1 Technical risks (and mitigations)
| Risk | Impact | Mitigation |
|---|---|---|
| **Lab→field domain gap** (PlantVillage 99% → field collapse) | headline result is fake if only in-distribution | mandatory cross-dataset test; segmentation; field data in eval |
| **Synthetic text leakage** | fake "multimodal gain" | symptom-only vocab, paraphrase, disjoint templates, text-only sanity ([§6.2](#62-the-leakage-problem-the-make-or-break-of-this-project)) |
| **Weather leakage / spurious signal** | inflated or meaningless weather branch | attach real weather by loc/date or label-independent climatology; ablate |
| **Few-shot overclaim** | reviewers reject "generalizes to new diseases" | strict novel-class split; ±CI over ≥600 episodes |
| **ViT on small data** | underperforms CNN | pretrained + freeze + regularize; keep CNN baseline |
| **Class imbalance** | accuracy hides rare-disease failure | macro-F1, weighted sampling, confusion matrix |
| **ASR errors (accents/noise/field)** | wrong symptom text | typed fallback; report WER separately; treat text as optional |
| **Open-set unknowns** | overconfident wrong diagnosis (safety) | reject below `θ`; "consult expert"; curated advice only |
| **Overconfident advice** | agronomic harm | curated KB not LLM; disclaimers; human-reviewed |

### 30.2 Computational bottlenecks
Heaviest → lightest: **SAM2** (GPU) > **Whisper large** > **ViT fine-tuning / episodic training** > Whisper small > **ViT frozen inference + prototype** (trivial). Design keeps the *deployed* path in the light zone; heavy items are training-time or research-only.

### 30.3 Requirements (ranges — assumptions stated, no fake exact times)
> Assumptions: ViT-B/16, ~tens-of-thousands of images, batch 32, 224²; "epoch" = one full pass. Times are **order-of-magnitude ranges**, hardware- and dataset-dependent — measure your own.

| Setup | Feasible? | RAM | Storage | Train (frozen ViT head) | Train (fine-tune last blocks) | Inference/req |
|---|---|---|---|---|---|---|
| **CPU-only laptop** | ✅ MVP train (frozen) + all inference | 8–16 GB | 20–60 GB (datasets dominate) | minutes–low hours/epoch | slow (hours+) — avoid | ViT+proto ~0.3–1 s; +Whisper small ~1–4 s |
| **Apple Silicon (M-series, MPS)** | ✅ good for MVP + light fine-tune | 16–32 GB | same | fast for frozen; OK for last-blocks | feasible for a few blocks | fast |
| **Consumer GPU (e.g., 8–12 GB)** | ✅ full research-grade | 16–32 GB | same | minutes/epoch | tens of min/epoch | very fast |
| **Cloud GPU (Colab/Kaggle/rented)** | ✅ heavy fine-tune, SAM2, large Whisper, sweeps | as provided | + dataset cache | fastest | fastest | n/a (train) |

**Runs locally comfortably:** ViT-B (frozen) inference, MiniLM/BERT text, Whisper `base`/`small`, `rembg`, prototype classifier, Approach-A/B fusion inference. **Prefer cloud for:** ViT fine-tuning at scale, SAM2, Whisper `large-v3`, long episodic meta-training, hyperparameter sweeps. **Storage:** models ~2–6 GB total; datasets are the main cost (finalized in [START HERE](#start-here)).

---

## 31. Final End-to-End Pipeline (inference)

```
User: leaf.jpg  +  (optional) speech "yellow spots and brown edges"  +  (optional) GPS
────────────────────────────────────────────────────────────────────────────────────
1. Validate image (mime, decode, min-res)                       → PIL RGB
2. (audio?) transcode→16kHz mono → Whisper small → symptoms_text  ["yellow spots..."]
   (no audio? use typed text; no text? text branch masked)
3. Preprocess image: rembg mask → crop leaf → resize 224² → normalize   [1,3,224,224]
4. ViT(frozen) → patch tokens [1,197,768] → pool → z_image [1,768] → proj [1,256]
5. Text: tokenizer → encoder → pool → z_text [1,H] → proj [1,256]   (or [1,L,256] tokens)
6. (GPS?) Weather API by lat/lon/date → [temp,humidity,rain,wind,...] → standardize(train μ,σ)
        → MLP → z_weather [1,256]      (no GPS? zeros + missing-mask)
7. Fusion:  MVP → concat[z_image,z_text,z_weather]→MLP→ z_fused [1,256]
            RG  → multimodal transformer over tokens → FUSION token → z_fused [1,256]
8. L2-normalize z_fused; cosine vs prototypes [C,256] → sim [C]
9. p = softmax(sim/τ); ŷ=argmax; conf=max(p); topk
10. if conf < θ_reject → {disease:null, low_confidence:true, message:"retake/consult"}
    else → disease key
11. RecommendationService: disease key → curated KB → {severity, actions, prevention, disclaimer}
12. Persist (diagnoses + weather_data); return JSON ([§23](#23-api-design))
────────────────────────────────────────────────────────────────────────────────────
→ { "disease":"rust", "confidence":0.92, "topk":[...], "weather":{...},
    "recommendations":[...], "disclaimer":"AI-assisted; confirm with an agronomist." }
```
**Missing-modality contract:** any of {text, weather} absent → its branch is masked/zeroed and fusion proceeds; image is the only hard requirement. Every response reports `modalities_used`.

---

## 32. Exact Next Steps (final implementation order)

Dependency-first. **Do this → verify this → then move on.** Never start meta-learning before the image baseline works.

| Step | Do this | Verify this before moving on |
|---|---|---|
| **1. Env** | create venv, install deps, detect device | `import torch` OK; device (CUDA/MPS/CPU) printed |
| **2. Dataset** | download primary (lab) + secondary (field); dedup; build `metadata.csv` | counts match source; grouped split; no near-dup across splits |
| **3. Preprocess** | rembg + crop + resize + normalize; save `data/interim` | 30 random cutouts look right; normalization matches ViT |
| **4. ViT baseline** | frozen ViT + head; train; evaluate in-dist **and cross-dataset** | sane macro-F1 in-dist; **record the field drop** (the anchor number) |
| **5. Voice** | Whisper small transcribe | WER acceptable on sample clips; typed fallback works |
| **6. Text encoder** | MiniLM embeddings + **text-only leakage check** | text-only accuracy moderate (NOT ~100%) |
| **7. Weather** | fetch/attach by lat/lon; MLP encoder | values in range; μ/σ from train only; missing-mask path |
| **8. Fusion** | Approach A (concat); then B (cross-attn) | A trains; B compared to A honestly (either result is fine) |
| **9. Prototypes** | build `prototypes.pt`; cosine classifier | cosine acc ≈ softmax head; add-new-class works |
| **10. Few-shot** | episodic ProtoNet; novel-class 1/5/10-shot | novel classes unseen in meta-train; report ±CI |
| **11. Backend** | FastAPI `/diagnose` runs pipeline (singleton models) | valid schema; low-confidence path fires |
| **12. Frontend** | React capture→result→history | camera/mic/geo + partial-input flows work |
| **13. Evaluation** | full ablation + cross-dataset + calibration | tables filled; ECE reported |
| **14. Deploy** | docker-compose (api+db+nginx); CPU inference | end-to-end demo from a phone browser |

---

## START HERE

**The first 5 concrete things to do right now.** All commands below use **verified** dataset slugs and model IDs (nothing fabricated). Steps 1–4 can be run today in order; step 5 is your first real result.

### 1) Environment (run today)
```bash
cd ~/Desktop/manishSirMLLeasfProject
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install torch torchvision transformers timm sentence-transformers \
            faster-whisper rembg albumentations opencv-python pillow \
            scikit-learn pandas imagehash fastapi "uvicorn[standard]" \
            python-multipart httpx pytest
python -c "import torch; print('cuda',torch.cuda.is_available(),'mps',torch.backends.mps.is_available())"
```

### 2) Create the project skeleton (run today)
```bash
mkdir -p data/{raw,interim,processed,prototypes} \
         ml/{data,preprocessing,models,training,inference,evaluation,services} \
         backend/{app,db,kb} frontend configs notebooks scripts tests docs
printf "data/raw/\ndata/interim/\ndata/processed/\n.venv/\n__pycache__/\n*.pt\n" > .gitignore
git init -q 2>/dev/null; echo "skeleton ready"
```

### 3) Download the datasets (verified slugs)
**Primary — PlantVillage** (lab, baseline + prototypes). Easiest via Hugging Face or TFDS (no Kaggle key needed):
```bash
# Option A — Hugging Face (no auth)
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("plant_village")          # ~815 MiB, 54,303 imgs, 38 classes
print(ds)
PY
# Option B — GitHub source (color/grayscale/segmented masks)
git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset data/raw/plantvillage
```
**Secondary — PlantDoc** (field, cross-dataset test; CC BY 4.0):
```bash
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset data/raw/plantdoc
```
**Few-shot / text — PlantWild** (in-the-wild, 89/115 classes, per-disease text; ⚠️ CC BY-NC-ND):
```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("uqtwei2/PlantWild")       # >18,000 imgs; ships text descriptions
print(ds)
PY
```
> Optional field-classifier primary (single crop): Cassava-2020 `kaggle competitions download -c cassava-leaf-disease-classification` (6.19 GB, needs `~/.kaggle/kaggle.json`), or Paddy Doctor `kaggle datasets download -d ... paddy-disease-classification`. **Confirm competition-rule licenses in [§4.3](#43-license-gate--read-before-commercial-use-or-redistribution) before any non-academic use.**

### 4) Download the models (verified IDs & licenses — see [§3](#3-recommended-technology-stack))
```python
# all licenses verified permissive (Apache-2.0 / MIT) in §3
from transformers import ViTModel, ViTImageProcessor
ViTModel.from_pretrained("google/vit-base-patch16-224")          # image encoder, 86.6M, Apache-2.0, 768-d
ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")    # text encoder, 384-d, Apache-2.0
from faster_whisper import WhisperModel; WhisperModel("small")   # ASR, 244M, downloads on first use
```

### 5) Run the first experiment: **ViT image baseline + cross-dataset test**
- Implement `ml/models/vit_encoder.py` (frozen ViT → 768 → head) and `ml/training/trainer.py`.
- Train on the **primary** dataset; evaluate on its test split **and** on the **secondary field** dataset.
- **Success = you can state two numbers:** in-distribution macro-F1, and the (expected large) drop on field data. That single honest result defines the problem your multimodal + few-shot work will then try to close — and it is the backbone everything else attaches to.

> After Step 5 works, proceed down [§32](#32-exact-next-steps-final-implementation-order) one row at a time. Resist jumping to meta-learning until the baseline + cross-dataset number exist.

---
*All research sections (§3 model table, §4/§5 datasets, §11 weather API, §29 literature) were completed from live web verification. Every dataset, model, API, and paper carries a primary-source link; anything not confirmable from a primary source is explicitly marked **UNVERIFIED**. No datasets, papers, APIs, models, or URLs were fabricated.*
