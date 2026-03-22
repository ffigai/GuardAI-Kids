"""Explanation utilities for model and policy decisions."""

from __future__ import annotations

import string

import numpy as np
import torch

from guardaikids.config import (
    CATEGORY_DESCRIPTIONS,
    IMAGE_SIMILARITY_PROMPTS,
    LABELS_ORDER,
    MAX_LENGTH,
    MODE,
    XAI_METHOD,
    get_default_thresholds,
)
from guardaikids.policy import get_policy_decision

SUPPORTED_XAI_METHODS = {"gradient_tokens"}


def clean_tokens(tokens: list[str]) -> list[str]:
    cleaned = []
    for token in tokens:
        if token[:1] in {"\u0120", "\u2581"}:
            cleaned.append(token[1:])
        elif token.startswith("##") and cleaned:
            cleaned[-1] = cleaned[-1] + token[2:]
        else:
            cleaned.append(token)
    return cleaned


def get_input_embeddings(model):
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise AttributeError("Model does not expose input embeddings for explanation generation.")
    return embeddings


def get_xai_method(method: str | None = None) -> str:
    selected = method or XAI_METHOD
    if selected not in SUPPORTED_XAI_METHODS:
        raise ValueError(f"Unsupported XAI method: {selected}")
    return selected


def explain_text_tokens(
    text: str,
    category_index: int,
    model,
    tokenizer,
    top_k: int = 5,
    image_features=None,
    xai_method: str | None = None,
):
    method = get_xai_method(xai_method)
    if method != "gradient_tokens":
        raise ValueError(f"Unsupported XAI method: {method}")

    model.eval()
    device = model.device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    embeddings = get_input_embeddings(model)(input_ids)
    embeddings.retain_grad()

    model_inputs = {
        "inputs_embeds": embeddings,
        "attention_mask": attention_mask,
    }
    if getattr(model, "mode", "text") == "multimodal":
        if image_features is None:
            raise ValueError("image_features are required to explain text tokens for multimodal mode.")
        image_tensor = torch.tensor(image_features, dtype=torch.float32, device=device).unsqueeze(0)
        model_inputs["image_features"] = image_tensor

    outputs = model(**model_inputs)
    target_probability = torch.sigmoid(outputs.logits)[0, category_index]

    model.zero_grad()
    target_probability.backward()

    token_importance = torch.norm(embeddings.grad[0], dim=1)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu())
    scored_tokens = list(zip(clean_tokens(tokens), token_importance.detach().cpu().numpy()))
    scored_tokens.sort(key=lambda item: item[1], reverse=True)

    return [
        item
        for item in scored_tokens
        if item[0] not in {"<s>", "</s>", "<pad>"} and item[0] not in string.punctuation and len(item[0]) > 2
    ][:top_k]


def format_age(age_group: str) -> str:
    return age_group.replace("_", "-")


def _short_prompt_label(prompt: str) -> str:
    lowered = prompt.lower()
    for phrase in [
        "addictive or substance-related content",
        "sexual or explicit content",
        "physical harm or self-injury",
        "hate speech or harassment",
        "weapons, threats, or intimidation",
        "disturbing or scary for children",
        "risky stunts, pranks, or dangerous challenges",
        "child-friendly harmless",
    ]:
        if phrase in lowered:
            return phrase
    return prompt.replace("a youtube thumbnail showing ", "").replace("a ", "")


def summarize_image_attributes(image_features, top_k: int = 3) -> list[dict[str, object]]:
    if image_features is None:
        return []
    vector = np.asarray(image_features, dtype=float)
    similarity_start = 512
    similarity_end = similarity_start + len(IMAGE_SIMILARITY_PROMPTS)

    if vector.shape[0] < similarity_end + 1:
        return []

    similarity_scores = vector[similarity_start:similarity_end]
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    return [
        {
            "type": "visual_prompt",
            "label": _short_prompt_label(IMAGE_SIMILARITY_PROMPTS[index]),
            "score": float(similarity_scores[index]),
        }
        for index in top_indices
    ]


def summarize_risk_categories(probs_row, age_group: str, mode: str, top_k: int = 3) -> list[dict[str, object]]:
    thresholds = get_default_thresholds(mode)
    warned = []
    for index, label in enumerate(LABELS_ORDER):
        probability = float(probs_row[index])
        warn_threshold = float(thresholds[age_group][label]["warn"])
        if probability >= warn_threshold:
            warned.append(
                {
                    "label": label,
                    "description": CATEGORY_DESCRIPTIONS[label],
                    "probability": probability,
                    "warn_threshold": warn_threshold,
                }
            )
    warned.sort(key=lambda item: item["probability"], reverse=True)
    if warned:
        return warned[:top_k]

    ranked = [
        {
            "label": label,
            "description": CATEGORY_DESCRIPTIONS[label],
            "probability": float(probs_row[index]),
            "warn_threshold": float(thresholds[age_group][label]["warn"]),
        }
        for index, label in enumerate(LABELS_ORDER)
    ]
    ranked.sort(key=lambda item: item["probability"], reverse=True)
    return ranked[:top_k]


def should_surface_supporting_cues(
    probs_row,
    age_group: str,
    mode: str,
    decision: str,
    margin: float = 0.03,
) -> bool:
    if decision != "Allow":
        return True
    thresholds = get_default_thresholds(mode)
    for index, label in enumerate(LABELS_ORDER):
        probability = float(probs_row[index])
        warn_threshold = float(thresholds[age_group][label]["warn"])
        if probability >= warn_threshold - margin:
            return True
    return False


def infer_modality_summary(mode: str, top_tokens, image_highlights) -> str:
    if mode == "text":
        return "Decision driven primarily by text evidence."
    if mode == "image":
        return "Decision driven primarily by thumbnail evidence."
    if top_tokens and image_highlights:
        return "Decision reflects both text and thumbnail evidence."
    if top_tokens:
        return "Decision driven mainly by text, with limited thumbnail contribution."
    if image_highlights:
        return "Decision driven mainly by the thumbnail, with limited textual contribution."
    return "Decision reflects combined model evidence."


def build_explanation_bullets(
    age_group: str,
    policy_info: dict[str, object],
    risk_categories: list[dict[str, object]],
    modality_summary: str,
    top_tokens,
    image_highlights,
    show_supporting_cues: bool,
) -> list[str]:
    bullets = [f"Decision: {policy_info['decision']} for ages {format_age(age_group)}."]

    if risk_categories and show_supporting_cues:
        category_text = "; ".join(
            f"{item['label']} ({item['probability']:.2f})" for item in risk_categories[:3]
        )
        bullets.append(f"Highest detected risk categories: {category_text}.")

    if policy_info["category"] is not None:
        bullets.append(
            f"Primary trigger: {policy_info['category']} at {policy_info['probability']:.2f} "
            f"against a threshold of {policy_info['threshold']:.2f}."
        )
    else:
        bullets.append("No category exceeded the current age-specific warning threshold.")

    if not show_supporting_cues:
        bullets.append("No text or thumbnail cue exceeded the current warning threshold.")
        return bullets

    bullets.append(modality_summary)

    if top_tokens:
        token_text = ", ".join(token for token, _ in top_tokens[:5])
        bullets.append(f"Text cues: {token_text}.")

    visual_cues = [item for item in image_highlights if item["type"] == "visual_prompt"]
    if visual_cues:
        cue_text = "; ".join(f"{item['label']} ({item['score']:.2f})" for item in visual_cues[:3])
        bullets.append(f"Thumbnail cues: {cue_text}.")

    return bullets


def format_bullets_as_text(bullets: list[str]) -> str:
    return "\n".join(f"- {bullet}" for bullet in bullets)


def explain_video(
    text: str,
    probs_row,
    age_group: str,
    model,
    tokenizer,
    thresholds=None,
    image_features=None,
    xai_method: str | None = None,
) -> dict[str, object]:
    mode = getattr(model, "mode", MODE)
    policy_info = get_policy_decision(
        probs_row,
        age_group,
        thresholds=thresholds,
        mode=mode,
    )
    risk_categories = summarize_risk_categories(probs_row, age_group, mode)
    image_highlights = summarize_image_attributes(image_features)

    top_tokens = []
    if mode != "image":
        chosen_category = policy_info["category"] or (risk_categories[0]["label"] if risk_categories else None)
        if chosen_category is not None:
            category_index = LABELS_ORDER.index(chosen_category)
            top_tokens = explain_text_tokens(
                text,
                category_index,
                model,
                tokenizer,
                image_features=image_features,
                xai_method=xai_method,
            )

    show_supporting_cues = should_surface_supporting_cues(
        probs_row,
        age_group=age_group,
        mode=mode,
        decision=policy_info["decision"],
    )
    if not show_supporting_cues:
        top_tokens = []
        image_highlights = []

    modality_summary = infer_modality_summary(mode, top_tokens, image_highlights)
    bullets = build_explanation_bullets(
        age_group=age_group,
        policy_info=policy_info,
        risk_categories=risk_categories,
        modality_summary=modality_summary,
        top_tokens=top_tokens,
        image_highlights=image_highlights,
        show_supporting_cues=show_supporting_cues,
    )

    return {
        "age_group": age_group,
        "decision": policy_info["decision"],
        "trigger_category": policy_info["category"],
        "probability": policy_info["probability"],
        "threshold": policy_info["threshold"],
        "top_tokens": top_tokens,
        "risk_categories": risk_categories,
        "image_highlights": image_highlights,
        "modality_summary": modality_summary,
        "explanation_bullets": bullets,
        "human_readable_explanation": format_bullets_as_text(bullets),
    }
