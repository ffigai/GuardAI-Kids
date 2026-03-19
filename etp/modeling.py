"""Model training, evaluation, and prediction helpers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, fbeta_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from etp.config import LABELS_ORDER, MAX_LENGTH, MODEL_NAME


@dataclass
class TrainingArtifacts:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    trainer: Trainer
    train_dataset: object
    val_dataset: object


def build_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataset(dataset, tokenizer, max_length: int = MAX_LENGTH):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def build_model(model_name: str = MODEL_NAME, num_labels: int = len(LABELS_ORDER)):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )


def load_saved_model(model_dir: str | Path, tokenizer_dir: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    return model, tokenizer


def build_training_args(output_dir: str = "./results", logging_dir: str = "./logs") -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=logging_dir,
        load_best_model_at_end=True,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
    }


def train_multilabel_classifier(train_dataset, val_dataset) -> TrainingArtifacts:
    tokenizer = build_tokenizer()
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_val = tokenize_dataset(val_dataset, tokenizer)
    model = build_model()
    trainer = Trainer(
        model=model,
        args=build_training_args(),
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


def predict_video_text(model, tokenizer, text: str, max_length: int = MAX_LENGTH) -> np.ndarray:
    model.eval()
    device = model.device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return torch.sigmoid(logits).cpu().numpy()[0]
