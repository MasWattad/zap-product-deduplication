import json
from openai import OpenAI


CONFIDENCE_MAP = {
    "high": 0.9,
    "medium": 0.6,
    "low": 0.3,
}


def _product_payload(product: dict) -> dict:
    return {
        "title": product.get("title"),
        "normalized_title": product.get("normalized_title"),
        "brand": product.get("brand"),
        "model_family": product.get("model_family"),
        "model_number": product.get("model_number"),
        "tier_variant": product.get("tier_variant"),
        "display_variant": product.get("display_variant"),
        "connectivity": product.get("connectivity"),
        "condition": product.get("condition"),
        "storage": product.get("storage"),
        "ram": product.get("ram"),
        "screen_size": product.get("screen_size"),
        "color": product.get("color"),
        "category": product.get("category"),
    }


def build_match_prompt(product_a: dict, product_b: dict, evidence: dict | None = None) -> str:
    payload = {
        "task": "product_match_decision",
        "prompt_version": "v4_evidence_routed",
        "instruction": (
            "Decide whether these two listings refer to the exact same sellable product. "
            "False merges are much worse than missed matches. "
            "Be conservative."
        ),
        "rules": [
            "Brand must match if present on both",
            "Model family must match if present on both",
            "Model number must match if present on both",
            "Storage must match if present on both",
            "Tier variant must match if present on both",
            "Display variant must match if present on both",
            "Connectivity must match if present on both",
            "Condition must match if present on both",
            "RAM must match if present on both",
            "Do not merge base/pro/ultra/fe variants",
            "Do not merge refurbished with new",
            "Do not merge GPS with GPS+Cellular",
            "Do not merge gen1 with gen2",
            "Do not merge only from semantic similarity",
            "If uncertain, return no",
            "Return only JSON"
        ],
        "product_a": _product_payload(product_a),
        "product_b": _product_payload(product_b),
        "evidence": evidence or {},
        "output_format": {
            "match": "yes_or_no",
            "confidence": "high_medium_low",
            "reason": "short explanation mentioning key matching or conflicting attributes"
        }
    }
    return json.dumps(payload, ensure_ascii=False)


def _extract_text(response) -> str:
    try:
        return response.output_text.strip()
    except Exception:
        pass

    try:
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


def _safe_parse_json(text: str):
    if not text:
        return None

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None

    return None


def _normalize_result(parsed: dict) -> dict:
    match_value = str(parsed.get("match", "no")).strip().lower()
    confidence_label = str(parsed.get("confidence", "low")).strip().lower()
    reason_value = str(parsed.get("reason", "missing_reason")).strip()

    if match_value not in {"yes", "no"}:
        match_value = "no"

    confidence_score = CONFIDENCE_MAP.get(confidence_label, 0.3)

    return {
        "match": match_value,
        "confidence": confidence_score,
        "reason": reason_value or "missing_reason",
    }


def ask_llm_match(product_a: dict, product_b: dict, model: str, client: OpenAI, evidence: dict | None = None) -> dict:
    prompt = build_match_prompt(product_a, product_b, evidence=evidence)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict ecommerce product matching assistant.\n"
                        "Return ONLY valid JSON.\n"
                        "Do NOT use markdown.\n"
                        "Do NOT add extra commentary.\n"
                        'Output EXACTLY: {"match":"yes_or_no","confidence":"high_medium_low","reason":"short explanation mentioning key matching or conflicting attributes"}'
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=140,
        )

        raw_text = _extract_text(response)
        parsed = _safe_parse_json(raw_text)

        if not parsed:
            return {
                "match": "no",
                "confidence": 0.3,
                "reason": "invalid_json_response",
            }

        return _normalize_result(parsed)

    except Exception as e:
        return {
            "match": "no",
            "confidence": 0.3,
            "reason": f"provider_error_{type(e).__name__}",
        }