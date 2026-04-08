# GuardAI Kids System Diagram

## High-Level Overview

```mermaid
%%{init: {'theme': 'default'}}%%
flowchart LR
    URL([YouTube URL])

    subgraph EXT[Data Extraction]
        E1[Title]
        E2[Description]
        E3[Channel Name]
        E4[Transcript]
        E5[Thumbnail URL]
    end

    subgraph PREP[Data Preparation]
        P1[Text Preparation]
        P2[Thumbnail Retrieval and Image Feature Extraction]
    end

    subgraph ANALYSIS[Data Analysis]
        A1[Text Analysis - DistilRoBERTa]
        A2[Multimodal Analysis - Text + Image Fusion MLP]
        A3[Image Analysis - CLIP + NSFW + Violence]
    end

    SCORES[Multi-Label Risk Scores - ADD, SXL, PH, HH]

    POLICY[Policy Decision - F2 Threshold Comparison]

    XAI[Explainability Layer]

    subgraph OUT[Final Output]
        O1[Safe / Harmful + Category Tags]
        O2[Text Evidence]
        O3[Image Evidence]
    end

    URL --> EXT
    E1 & E2 & E3 & E4 --> P1
    E5 --> P2

    P1 --> A1
    P2 --> A3
    A1 --> A2
    A3 --> A2

    A1 & A2 & A3 --> SCORES

    SCORES --> POLICY
    SCORES --> XAI

    POLICY --> O1
    XAI --> O2
    XAI --> O3
```

## Full Pipeline

```mermaid
flowchart TD
    A[MetaHarm Excel Datasets<br/>Harmful.xlsx + Harmless.xlsx] --> B[Data Loading<br/>load_raw_data]
    B --> C[Label Encoding + Text Assembly<br/>prepare_model_dataframe]
    C --> D[Train/Validation Split<br/>split_train_validation]

    A --> T1[Thumbnail Download Script<br/>scripts/fetch_dataset_thumbnails.py]
    T1 --> T2[data/thumbnails]

    T2 --> T3[Image Feature Extraction<br/>scripts/extract_image_features.py]

    subgraph T3_detail[Image Feature Extraction - clip_nsfw_violence]
        TA[CLIP Image Encoder<br/>512-dim visual embedding]
        TB[CLIP Text Encoder<br/>8 harm-prompt similarity scores]
        TC[Quality Features<br/>brightness, contrast, saturation,<br/>colorfulness, detail]
        TD_nsfw[Marqo NSFW Classifier<br/>NSFW probability score]
        TE_viol[Violence Classifier ViT-B/16<br/>violence probability score]
        TF[EasyOCR -> CLIP<br/>8 OCR harm-similarity scores + flag]
    end

    T2 --> TA
    T2 --> TB
    T2 --> TC
    T2 --> TD_nsfw
    T2 --> TE_viol
    T2 --> TF
    TA --> T4
    TB --> T4
    TC --> T4
    TD_nsfw --> T4
    TE_viol --> T4
    TF --> T4

    T4[data/image_features/*.npy<br/>537-dim feature vectors]

    D --> E[Hugging Face Dataset Conversion<br/>to_hf_dataset]
    T4 --> F[Mode-Aware Input Preparation<br/>prepare_dataset_inputs]
    E --> F

    F --> G{Mode}
    G -->|Text| H1[Tokenization<br/>input_ids + attention_mask]
    G -->|Image| H2[Load 537-dim image_features]
    G -->|Multimodal| H3[Tokenization + 537-dim image_features]

    H1 --> I1[Text Model<br/>DistilRoBERTa + text head]
    H2 --> I2[Image Model<br/>MLP on 537-dim image features]
    H3 --> I3[Multimodal Model<br/>Text encoder + fusion MLP]

    I1 --> J[Trainer Training + Validation]
    I2 --> J
    I3 --> J

    J --> K[Validation Prediction<br/>trainer.predict]
    K --> L[Sigmoid Probabilities]
    L --> M[Model Metrics<br/>macro F1, micro F1, ROC-AUC]
    L --> N[Policy Layer<br/>Safe / Harmful + category tags]
    N --> O[Policy Metrics<br/>precision, recall, false positive rate, protection rate]

    J --> P[Artifacts<br/>artifacts/text<br/>artifacts/image_clip_nsfw_violence<br/>artifacts/multimodal_clip_nsfw_violence]
    P --> Q[metadata.json<br/>incl. f2_thresholds]
    P --> R[predictions_mode.json]
    P --> S[model + tokenizer]

    R --> U[Policy Reevaluation Script]
    R --> V[Experiment Report Script]
    S --> W[Explainability Layer]
    Q --> V

    W --> X[Text token attributions<br/>gradient_tokens]
    W --> Y[Image cue summaries<br/>incl. OCR text cues]
    X --> Z[Web UI / CLI Output]
    Y --> Z
    N --> Z
```

## Runtime Analysis Flow

```mermaid
flowchart LR
    A[YouTube URL] --> B

    subgraph B[Data Extraction]
        B1[Title]
        B2[Description]
        B3[Channel Name]
        B4[Transcript]
        B5[Thumbnail URL]
    end

    subgraph C[Data Preparation]
        C1[Text Preparation]
        C2[Thumbnail Retrieval]
    end

    B1 --> C1
    B2 --> C1
    B3 --> C1
    B4 --> C1
    B5 --> C2

    subgraph D[Data Analysis]
        D1[Text Analysis<br/>DistilRoBERTa]

        subgraph D2[Image Analysis - clip_nsfw_violence]
            D2a[CLIP visual embedding<br/>512-dim]
            D2b[CLIP harm-prompt similarity<br/>8 scores]
            D2c[Quality features<br/>5 scores]
            D2d[Marqo NSFW score]
            D2e[Violence ViT-B/16 score]
            D2f[EasyOCR -> CLIP<br/>8 OCR scores + flag]
        end

        D3[Multimodal Analysis<br/>Text + Image fusion MLP]
        D4[Multi-Label Risk Scores<br/>ADD - SXL - PH - HH]
    end

    C1 --> D1
    C2 --> D2a
    C2 --> D2b
    C2 --> D2c
    C2 --> D2d
    C2 --> D2e
    C2 --> D2f
    C1 --> D3
    C2 --> D3

    D1 --> D4
    D2a --> D4
    D2b --> D4
    D2c --> D4
    D2d --> D4
    D2e --> D4
    D2f --> D4
    D3 --> D4

    subgraph E[Policy Decision]
        E1[F2 Threshold Comparison]
        E2[Safe / Harmful + category tags]
    end

    D4 --> E1
    E1 --> E2

    subgraph F[Explainability]
        F1[Explanation Layer]
        F2[Text Evidence<br/>gradient token attribution]
        F3[Image Evidence<br/>visual + OCR text cues]
    end

    D4 --> F1
    E1 --> F1
    F1 --> F2
    F1 --> F3

    G[Final Output]

    E2 --> G
    F2 --> G
    F3 --> G
```

## Component View

```mermaid
flowchart TB
    subgraph DataPrep[Dataset Preparation]
        A1[fetch_dataset_thumbnails.py]
        A2[extract_image_features.py<br/>clip_nsfw_violence mode]
        A3[backfill_thumbnail_labels.py]
    end

    subgraph Core[Core Package: src/guardaikids]
        B1[data.py]
        B2[modeling.py]
        B3[policy.py]
        B4[workflow.py]
        B5[service.py]
        B6[explainability.py]
        B7[image_features.py<br/>CLIP + NSFW + violence + OCR feature builders]
        B8[youtube.py]
        B9[web_interface.py]
        B10[main.py]
    end

    subgraph Outputs[Outputs]
        C1[artifacts/text]
        C2[artifacts/image_clip_nsfw_violence]
        C3[artifacts/multimodal_clip_nsfw_violence]
        C4[artifacts/reports]
    end

    DataPrep --> Core
    Core --> Outputs
```

## Image Feature Vector Layout (clip_nsfw_violence)

| Dimensions | Source | Model |
|---|---|---|
| 0-511 | Visual embedding | CLIP ViT-B/32 image encoder |
| 512-519 | Harm-prompt similarity (visual) | CLIP text encoder x 8 prompts |
| 520-524 | Quality features | Handcrafted (brightness, contrast, saturation, colorfulness, detail) |
| 525 | NSFW score | Marqo/nsfw-image-detection-384 |
| 526 | Violence score | jaranohaal/vit-base-violence-detection (ViT-B/16) |
| 527 | has_ocr_text flag | EasyOCR gate |
| 528-535 | Harm-prompt similarity (OCR text) | CLIP text encoder x 8 prompts |
| 536 | Missing image flag | - |

**Total: 537-dim**

## Notes

- Text mode uses `distilroberta-base` with CLS-token style pooling.
- Image and multimodal modes use precomputed 537-dim thumbnail features (`clip_nsfw_violence` backend).
- Multimodal fusion concatenates the 768-dim text embedding with the 537-dim image vector, then applies an MLP fusion head.
- Decision thresholds are F2-optimized per label per mode and stored in each artifact's `metadata.json`. They are loaded at runtime - no hardcoded thresholds.
- Explainability is gradient-based for text tokens and cue-summary-based for image features (including OCR text cues).
