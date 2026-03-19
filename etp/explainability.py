"""Explanation utilities for model and policy decisions."""

import string

import torch

from etp.config import CATEGORY_DESCRIPTIONS, LABELS_ORDER, MAX_LENGTH
from etp.policy import get_policy_decision


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


def explain_text_tokens(text: str, category_index: int, model, tokenizer, top_k: int = 5):
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

    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
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


def generate_human_explanation(age_group: str, policy_info: dict[str, object], top_tokens) -> str:
    decision = policy_info["decision"]
    formatted_age = format_age(age_group)
    if decision == "Allow":
        return (
            f"This content is considered appropriate for age group {formatted_age}. "
            "No significant harmful indicators were detected."
        )

    category = policy_info["category"]
    probability = policy_info["probability"]
    threshold = policy_info["threshold"]
    description = CATEGORY_DESCRIPTIONS.get(category, "harmful content")
    token_words = ", ".join(token for token, _ in top_tokens) if top_tokens else "no dominant terms"

    if decision == "Block":
        return (
            f"This content was blocked for age group {formatted_age} because it shows a high likelihood of "
            f"{description}. The predicted risk score ({probability:.2f}) exceeded the safety threshold "
            f"({threshold:.2f}). Key terms influencing this decision include: {token_words}."
        )

    return (
        f"This content triggered a warning for age group {formatted_age} due to potential {description}. "
        f"The predicted risk score ({probability:.2f}) exceeded the caution threshold ({threshold:.2f}). "
        f"Detected terms contributing to this assessment include: {token_words}."
    )


def explain_video(text: str, probs_row, age_group: str, model, tokenizer, thresholds=None) -> dict[str, object]:
    policy_info = get_policy_decision(probs_row, age_group, thresholds=thresholds)
    if policy_info["category"] is not None:
        category_index = LABELS_ORDER.index(policy_info["category"])
        top_tokens = explain_text_tokens(text, category_index, model, tokenizer)
    else:
        top_tokens = []

    return {
        "age_group": age_group,
        "decision": policy_info["decision"],
        "trigger_category": policy_info["category"],
        "probability": policy_info["probability"],
        "threshold": policy_info["threshold"],
        "top_tokens": top_tokens,
        "human_readable_explanation": generate_human_explanation(age_group, policy_info, top_tokens),
    }
