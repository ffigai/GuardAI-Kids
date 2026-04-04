"""Model training, evaluation, and prediction helpers."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, fbeta_score, roc_auc_score
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from guardaikids.config import IMAGE_ANALYSIS_MODEL, IMAGE_FEATURE_DIMS, LABELS_ORDER, MAX_LENGTH, MODE, MODEL_NAME
from guardaikids.data import prepare_dataset_inputs

CUSTOM_CONFIG_NAME = "guardaikids_model_config.json"
CUSTOM_WEIGHTS_NAME = "guardaikids_model.bin"


@dataclass
class TrainingArtifacts:
    tokenizer: AutoTokenizer
    model: nn.Module
    trainer: Trainer
    train_dataset: object
    val_dataset: object
    mode: str


class MultimodalSequenceClassifier(nn.Module):
    """A Trainer-compatible multi-label classifier supporting text, image, and fused inputs."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = len(LABELS_ORDER),
        mode: str = MODE,
        image_feature_dim: int = IMAGE_FEATURE_DIMS[IMAGE_ANALYSIS_MODEL],
        fusion_hidden_dim: int | None = None,
        text_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"text", "image", "multimodal"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.model_name = model_name
        self.mode = mode
        self.num_labels = num_labels
        self.image_feature_dim = image_feature_dim

        if text_encoder is not None:
            self.text_encoder = text_encoder
            text_hidden_size = text_encoder.config.hidden_size
        elif mode in {"text", "multimodal"}:
            self.text_encoder = AutoModel.from_pretrained(model_name)
            text_hidden_size = self.text_encoder.config.hidden_size
        else:
            self.text_encoder = None
            text_hidden_size = AutoConfig.from_pretrained(model_name).hidden_size

        self.text_hidden_size = text_hidden_size
        self.fusion_hidden_dim = fusion_hidden_dim or text_hidden_size
        self.image_hidden_dim = max(128, min(512, image_feature_dim // 2))
        self.text_classifier = nn.Linear(text_hidden_size, num_labels)
        self.image_classifier = nn.Sequential(
            nn.LayerNorm(image_feature_dim),
            nn.Linear(image_feature_dim, self.image_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(self.image_hidden_dim, num_labels),
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(text_hidden_size + image_feature_dim, self.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.fusion_hidden_dim, num_labels),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_embeddings(self):
        if self.text_encoder is None:
            return None
        return self.text_encoder.get_input_embeddings()

    def _get_text_embedding(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.text_encoder is None:
            raise ValueError(f"Text embeddings are unavailable when mode={self.mode}.")
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        return outputs.last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ) -> SequenceClassifierOutput:
        if self.mode == "text":
            text_embedding = self._get_text_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            logits = self.text_classifier(text_embedding)
        elif self.mode == "image":
            if image_features is None:
                raise ValueError("image_features are required when mode='image'.")
            logits = self.image_classifier(image_features.float())
        else:
            if image_features is None:
                raise ValueError("image_features are required when mode='multimodal'.")
            text_embedding = self._get_text_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            combined = torch.cat([text_embedding, image_features.float()], dim=-1)
            logits = self.fusion_layer(combined)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def save_pretrained(self, save_directory: str | Path) -> None:
        resolved_dir = Path(save_directory)
        resolved_dir.mkdir(parents=True, exist_ok=True)
        if self.text_encoder is not None:
            self.text_encoder.save_pretrained(resolved_dir / "text_encoder")
        torch.save(self.state_dict(), resolved_dir / CUSTOM_WEIGHTS_NAME)
        config = {
            "model_name": self.model_name,
            "mode": self.mode,
            "num_labels": self.num_labels,
            "image_feature_dim": self.image_feature_dim,
            "fusion_hidden_dim": self.fusion_hidden_dim,
        }
        with (resolved_dir / CUSTOM_CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

    @classmethod
    def from_pretrained(cls, model_directory: str | Path):
        model_dir = Path(model_directory)
        with (model_dir / CUSTOM_CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        text_encoder = None
        text_encoder_dir = model_dir / "text_encoder"
        if text_encoder_dir.exists():
            text_encoder = AutoModel.from_pretrained(str(text_encoder_dir))

        model = cls(
            model_name=config["model_name"],
            num_labels=config["num_labels"],
            mode=config["mode"],
            image_feature_dim=config["image_feature_dim"],
            fusion_hidden_dim=config["fusion_hidden_dim"],
            text_encoder=text_encoder,
        )
        state_dict = torch.load(model_dir / CUSTOM_WEIGHTS_NAME, map_location="cpu")
        # pos_weight is a training-only buffer; strip it so strict loading works at inference time.
        state_dict.pop("loss_fn.pos_weight", None)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            old_linear_image_head = {"image_classifier.weight", "image_classifier.bias"}.issubset(state_dict.keys())
            if model.mode == "text" and old_linear_image_head:
                filtered_state_dict = {
                    key: value for key, value in state_dict.items() if not key.startswith("image_classifier.")
                }
                missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
                warnings.warn(
                    "Loaded a legacy text-mode artifact with an older image head layout. "
                    f"Ignored image head weights. Missing: {missing}. Unexpected: {unexpected}.",
                    RuntimeWarning,
                )
            else:
                raise RuntimeError(
                    "Saved artifact is incompatible with the current model architecture. "
                    "Text artifacts from before the image-head refactor can be reloaded automatically, "
                    "but image or multimodal artifacts saved with the older 518-d image pipeline "
                    "must be retrained with the current code."
                ) from exc
        return model


def build_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataset(dataset, tokenizer, max_length: int = MAX_LENGTH, mode: str = MODE):
    if mode in {"text", "multimodal"}:
        def tokenize(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        tokenized = dataset.map(tokenize, batched=True)
    else:
        tokenized = dataset

    columns = ["labels"]
    if mode in {"text", "multimodal"}:
        columns = ["input_ids", "attention_mask"] + columns
    if mode in {"image", "multimodal"}:
        columns = ["image_features"] + columns
    tokenized.set_format(type="torch", columns=columns)
    return tokenized


def build_model(
    model_name: str = MODEL_NAME,
    num_labels: int = len(LABELS_ORDER),
    mode: str = MODE,
    image_feature_dim: int | None = None,
):
    return MultimodalSequenceClassifier(
        model_name=model_name,
        num_labels=num_labels,
        mode=mode,
        image_feature_dim=image_feature_dim or IMAGE_FEATURE_DIMS[IMAGE_ANALYSIS_MODEL],
    )


def load_saved_model(model_dir: str | Path, tokenizer_dir: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model = MultimodalSequenceClassifier.from_pretrained(model_dir)
    return model, tokenizer


def build_training_args(output_dir: str = "./results", mode: str = MODE) -> TrainingArguments:
    # Image-only mode trains a randomly initialised MLP — needs a higher LR and more epochs.
    # Text/multimodal modes fine-tune a pretrained transformer, so keep the conservative LR.
    if mode == "image":
        lr = 1e-3
        epochs = 15
    else:
        lr = 2e-5
        epochs = 3
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
    }


def train_multilabel_classifier(
    train_dataset,
    val_dataset,
    mode: str = MODE,
    image_feature_dim: int | None = None,
    image_feature_dir: str | None = None,
) -> TrainingArtifacts:
    tokenizer = build_tokenizer()
    prepared_train = prepare_dataset_inputs(train_dataset, mode=mode, image_feature_dir=image_feature_dir, image_feature_dim=image_feature_dim)
    prepared_val = prepare_dataset_inputs(val_dataset, mode=mode, image_feature_dir=image_feature_dir, image_feature_dim=image_feature_dim)
    tokenized_train = tokenize_dataset(prepared_train, tokenizer, mode=mode)
    tokenized_val = tokenize_dataset(prepared_val, tokenizer, mode=mode)
    model = build_model(mode=mode, image_feature_dim=image_feature_dim)

    # Compute per-label pos_weight = (N - n_pos) / n_pos, capped at 10 to avoid over-prediction.
    all_labels = np.array(tokenized_train["labels"])
    n_pos = all_labels.sum(axis=0).clip(min=1)
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor((n_neg / n_pos).clip(max=10.0), dtype=torch.float32).to(model.device)
    model.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    trainer = Trainer(
        model=model,
        args=build_training_args(mode=mode),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return TrainingArtifacts(
        tokenizer=tokenizer,
        model=model,
        trainer=trainer,
        train_dataset=tokenized_train,
        val_dataset=tokenized_val,
        mode=mode,
    )


def collect_validation_outputs(trainer: Trainer, val_dataset) -> dict[str, np.ndarray]:
    outputs = trainer.predict(val_dataset)
    logits = torch.tensor(outputs.predictions)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    labels = outputs.label_ids
    return {
        "logits": outputs.predictions,
        "probs": probs,
        "preds": preds,
        "labels": labels,
    }


def summarize_validation_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> dict[str, object]:
    report = classification_report(labels, preds, target_names=LABELS_ORDER, zero_division=0)
    roc_auc = {}
    for index, label in enumerate(LABELS_ORDER):
        try:
            roc_auc[label] = roc_auc_score(labels[:, index], probs[:, index])
        except ValueError:
            roc_auc[label] = None
    return {
        "classification_report": report,
        "score_distribution": pd.DataFrame(probs, columns=LABELS_ORDER).describe(),
        "roc_auc": roc_auc,
        "positives": {label: int(labels[:, idx].sum()) for idx, label in enumerate(LABELS_ORDER)},
    }


def optimize_thresholds(labels: np.ndarray, probs: np.ndarray, beta: float = 1.0) -> dict[str, float]:
    optimal_thresholds = {}
    for index, label in enumerate(LABELS_ORDER):
        best_score = -1.0
        best_threshold = 0.5
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (probs[:, index] > threshold).astype(int)
            if beta == 1.0:
                score = f1_score(labels[:, index], preds, zero_division=0)
            else:
                score = fbeta_score(labels[:, index], preds, beta=beta, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        optimal_thresholds[label] = best_threshold
    return optimal_thresholds


def apply_thresholds(probs: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    predictions = np.zeros_like(probs)
    for index, label in enumerate(LABELS_ORDER):
        predictions[:, index] = (probs[:, index] > thresholds[label]).astype(int)
    return predictions


def predict_video_text(
    model,
    tokenizer,
    text: str,
    max_length: int = MAX_LENGTH,
    image_features: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    model.eval()
    device = model.device
    input_kwargs: dict[str, torch.Tensor] = {}

    if model.mode in {"text", "multimodal"}:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        input_kwargs["input_ids"] = inputs["input_ids"].to(device)
        input_kwargs["attention_mask"] = inputs["attention_mask"].to(device)

    if model.mode in {"image", "multimodal"}:
        if image_features is None:
            raise ValueError(f"image_features are required when mode='{model.mode}'.")
        if len(image_features) != model.image_feature_dim:
            raise ValueError(
                f"Artifact expects image feature vectors of length {model.image_feature_dim}, "
                f"but received {len(image_features)}. Retrain the {model.mode} artifact with the current code."
            )
        image_tensor = torch.tensor(image_features, dtype=torch.float32, device=device).unsqueeze(0)
        input_kwargs["image_features"] = image_tensor

    with torch.no_grad():
        outputs = model(**input_kwargs)
        logits = outputs.logits
        return torch.sigmoid(logits).cpu().numpy()[0]
