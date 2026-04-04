# GuardAI Kids System Diagram

## Full Pipeline

```mermaid
flowchart TD
    A[MetaHarm Excel Datasets<br/>Harmful.xlsx + Harmless.xlsx] --> B[Data Loading<br/>load_raw_data]
    B --> C[Label Encoding + Text Assembly<br/>prepare_model_dataframe]
    C --> D[Train/Validation Split<br/>split_train_validation]

    A --> T1[Thumbnail Download Script<br/>scripts/fetch_dataset_thumbnails.py]
    T1 --> T2[data/thumbnails]

    T2 --> T3[Image Feature Extraction<br/>scripts/extract_image_features.py]

    subgraph T3_detail[Image Feature Extraction — clip_ocr mode]
        TA[CLIP Image Encoder<br/>512-dim visual embedding]
        TB[CLIP Text Encoder<br/>8 harm-prompt similarity scores]
        TC[Quality Features<br/>brightness, contrast, saturation,<br/>colorfulness, detail]
        TD_ocr[EasyOCR<br/>reads words from thumbnail]
        TE[CLIP Text Encoder<br/>OCR text → 8 harm-prompt scores]
        TF[has_ocr_text flag]
    end

    T2 --> TA
    T2 --> TB
    T2 --> TC
    T2 --> TD_ocr
    TD_ocr --> TE
    TD_ocr --> TF
    TA --> T4
    TB --> T4
    TC --> T4
    TE --> T4
    TF --> T4

    T4[data/image_features/*.npy<br/>535-dim feature vectors]

    D --> E[Hugging Face Dataset Conversion<br/>to_hf_dataset]
    T4 --> F[Mode-Aware Input Preparation<br/>prepare_dataset_inputs]
    E --> F

    F --> G{Mode}
    G -->|Text| H1[Tokenization<br/>input_ids + attention_mask]
    G -->|Image| H2[Load 535-d image_features]
    G -->|Multimodal| H3[Tokenization + 535-d image_features]

    H1 --> I1[Text Model<br/>DistilRoBERTa + text head]
    H2 --> I2[Image Model<br/>MLP on 535-d image features]
    H3 --> I3[Multimodal Model<br/>Text encoder + fusion MLP]

    I1 --> J[Trainer Training + Validation]
    I2 --> J
    I3 --> J

    J --> K[Validation Prediction<br/>trainer.predict]
    K --> L[Sigmoid Probabilities]
    L --> M[Model Metrics<br/>macro F1, micro F1, ROC-AUC]
    L --> N[Age Policy Layer<br/>Allow / Warn / Block]
    N --> O[Policy Metrics<br/>precision, recall, false allow, protection]

    J --> P[Artifacts per Mode<br/>artifacts/text<br/>artifacts/image<br/>artifacts/multimodal]
    P --> Q[metadata.json]
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

        subgraph D2[Image Analysis — clip_ocr]
            D2a[CLIP visual embedding<br/>512-dim]
            D2b[CLIP harm-prompt similarity<br/>8 scores]
            D2c[Quality features<br/>5 scores]
            D2d[EasyOCR text extraction]
            D2e[CLIP OCR text similarity<br/>8 scores]
            D2f[has_ocr_text flag]
        end

        D3[Multimodal Analysis<br/>Text + Image fusion MLP]
        D4[Multi-Label Risk Scores<br/>ADD · SXL · PH · HH]
    end

    C1 --> D1
    C2 --> D2a
    C2 --> D2b
    C2 --> D2c
    C2 --> D2d
    D2d --> D2e
    D2d --> D2f
    C1 --> D3
    C2 --> D3

    D1 --> D4
    D2a --> D4
    D2b --> D4
    D2c --> D4
    D2e --> D4
    D2f --> D4
    D3 --> D4

    subgraph E[Age-Aware Policy]
        E1[Policy Decision]
        E2[Allow / Warn / Block by Age Group]
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
        A2[extract_image_features.py<br/>clip or clip_ocr mode]
        A3[backfill_thumbnail_labels.py]
    end

    subgraph Core[Core Package: src/guardaikids]
        B1[data.py]
        B2[modeling.py]
        B3[policy.py]
        B4[workflow.py]
        B5[service.py]
        B6[explainability.py]
        B7[image_features.py<br/>clip + clip_ocr feature builders<br/>EasyOCR integration]
        B8[youtube.py]
        B9[web_interface.py]
        B10[main.py]
    end

    subgraph Outputs[Outputs]
        C1[artifacts/text]
        C2[artifacts/image]
        C3[artifacts/multimodal]
        C4[artifacts/reports]
    end

    DataPrep --> Core
    Core --> Outputs
```

## Image Feature Vector Layout

| Dimensions | Source | Model |
|---|---|---|
| 0–511 | Visual embedding | CLIP image encoder |
| 512–519 | Harm-prompt similarity (visual) | CLIP text encoder × 8 prompts |
| 520–524 | Quality features | Handcrafted (brightness, contrast, saturation, colorfulness, detail) |
| 525 | Missing image flag | — |
| 526–533 | Harm-prompt similarity (OCR text) | CLIP text encoder × 8 prompts `clip_ocr only` |
| 534 | has_ocr_text flag | EasyOCR `clip_ocr only` |

**Total: 526-dim (`clip`) · 535-dim (`clip_ocr`)**

## Notes

- Text mode uses `distilroberta-base` with CLS-token style pooling.
- Image mode uses precomputed `535`-dim thumbnail features when `IMAGE_ANALYSIS_MODEL = "clip_ocr"`.
- `clip_ocr` adds EasyOCR text extraction on top of CLIP visual features. OCR text is encoded by the same CLIP text encoder already loaded, then compared against the same 8 harm prompts.
- OCR text is truncated to CLIP's 77-token limit before encoding.
- Multimodal mode concatenates text embedding and image features, then applies an MLP fusion head.
- Policy decisions are separate from model training and can be reevaluated from saved predictions.
- Explainability is currently gradient-based for text and cue-summary-based for image (including OCR text cues).
- Switching back to `clip` (526-dim) requires changing `IMAGE_ANALYSIS_MODEL` in `config.py` and retraining.
