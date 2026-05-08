import re

_VALID_LABELS = frozenset({"A", "B", "C", "D"})
_DIRECT_LABEL_PATTERN = re.compile(r"""^[\s"'`([{]*([A-D])[\s"'`)\].,:;!?]*$""", re.IGNORECASE)
_XML_ANSWER_PATTERN = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE | re.DOTALL)


def _normalize_label(text: str) -> str | None:
    match = _DIRECT_LABEL_PATTERN.fullmatch(text)
    if match is None:
        return None
    label = match.group(1).upper()
    if label not in _VALID_LABELS:
        return None
    return label


def extract_xml_answer(text: str) -> str | None:
    """Extract the option label from the legacy XML answer format."""
    match = _XML_ANSWER_PATTERN.search(text)
    if match is None:
        return None
    return match.group(1).upper()


def _extract_prediction(solution: str) -> tuple[str | None, str]:
    prediction = _normalize_label(solution.strip())
    if prediction is not None:
        return prediction, "direct_label"

    prediction = extract_xml_answer(solution)
    if prediction is not None:
        return prediction, "legacy_xml"

    return None, "none"


def compute_score(solution: str, ground_truth: str) -> dict:
    prediction, parser_mode = _extract_prediction(solution)
    normalized_ground_truth = _normalize_label(str(ground_truth).strip()) or str(ground_truth).strip().upper()
    reward = float(prediction == normalized_ground_truth)

    if reward == 1.0:
        feedback = ""
    elif prediction is None:
        feedback = "The response must be a single option label (A, B, C, or D)."
    else:
        feedback = "The option label is incorrect."

    incorrect_format = 0.0 if parser_mode == "direct_label" else 1.0
    return {
        "score": reward,
        "acc": reward,
        "pred": prediction,
        "incorrect_format": incorrect_format,
        "feedback": feedback,
        "parser_mode": parser_mode,
    }
