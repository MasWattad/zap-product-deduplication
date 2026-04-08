from collections import defaultdict
from itertools import combinations

import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity


SPECIAL_TIER_VARIANTS = {"pro", "ultra", "plus", "fe"}


def build_block_keys(row) -> set[str]:
    keys = set()

    brand = row.get("brand")
    model_family = row.get("model_family")
    model_number = row.get("model_number")
    tier_variant = row.get("tier_variant")
    display_variant = row.get("display_variant")
    connectivity = row.get("connectivity")
    storage = row.get("storage")
    ram = row.get("ram")
    condition = row.get("condition")
    screen_size = row.get("screen_size")
    normalized_title = row.get("normalized_title")

    if brand and model_family and model_number:
        keys.add(f"core|{brand}|{model_family}|{model_number}")

    if brand and model_number and storage:
        keys.add(f"core_storage|{brand}|{model_number}|{storage}")

    if brand and model_family and storage:
        keys.add(f"family_storage|{brand}|{model_family}|{storage}")

    if brand and model_family and tier_variant:
        keys.add(f"family_variant|{brand}|{model_family}|{tier_variant}")

    if brand and model_family and display_variant:
        keys.add(f"family_display|{brand}|{model_family}|{display_variant}")

    if brand and model_family and connectivity:
        keys.add(f"family_conn|{brand}|{model_family}|{connectivity}")

    if brand and model_family and ram:
        keys.add(f"family_ram|{brand}|{model_family}|{ram}")

    if brand and model_family and screen_size:
        keys.add(f"family_size|{brand}|{model_family}|{screen_size}")

    if brand and model_family and condition:
        keys.add(f"family_condition|{brand}|{model_family}|{condition}")

    if brand and model_family:
        keys.add(f"family|{brand}|{model_family}")
        if model_number:
            keys.add(f"loose|{brand}|{model_family}|{model_number}")
        else:
            keys.add(f"loose|{brand}|{model_family}")

    if not keys and normalized_title:
        tokens = normalized_title.split()
        if len(tokens) >= 2:
            keys.add(f"fallback|{tokens[0]}|{tokens[1]}")

    return keys


def build_block_index(df: pd.DataFrame) -> dict:
    block_index = defaultdict(list)
    for _, row in df.iterrows():
        for key in row["block_keys"]:
            block_index[key].append(row["listing_id"])
    return block_index


def generate_candidate_pairs(block_index: dict) -> set[tuple]:
    candidate_pairs = set()
    for _, listing_ids in block_index.items():
        unique_ids = sorted(set(listing_ids))
        if len(unique_ids) < 2:
            continue
        for a, b in combinations(unique_ids, 2):
            candidate_pairs.add((a, b))
    return candidate_pairs


def _same_core_identity(row_a, row_b) -> bool:
    return (
        row_a.get("brand") == row_b.get("brand")
        and row_a.get("model_family") == row_b.get("model_family")
        and row_a.get("model_number") == row_b.get("model_number")
    )


def has_hard_conflict(row_a, row_b) -> tuple[bool, str]:
    if row_a.get("brand") and row_b.get("brand") and row_a["brand"] != row_b["brand"]:
        return True, "brand_mismatch"

    if row_a.get("model_family") and row_b.get("model_family") and row_a["model_family"] != row_b["model_family"]:
        return True, "model_family_mismatch"

    if row_a.get("model_number") and row_b.get("model_number") and row_a["model_number"] != row_b["model_number"]:
        return True, "model_number_mismatch"

    if row_a.get("storage") and row_b.get("storage") and row_a["storage"] != row_b["storage"]:
        return True, "storage_mismatch"

    var_a = row_a.get("tier_variant")
    var_b = row_b.get("tier_variant")

    # Hard reject only when BOTH sides explicitly have different variants
    if var_a is not None and var_b is not None and var_a != var_b:
        return True, "tier_variant_mismatch"

    if row_a.get("display_variant") and row_b.get("display_variant") and row_a["display_variant"] != row_b["display_variant"]:
        return True, "display_variant_mismatch"

    if row_a.get("connectivity") and row_b.get("connectivity") and row_a["connectivity"] != row_b["connectivity"]:
        return True, "connectivity_mismatch"

    if row_a.get("ram") and row_b.get("ram") and row_a["ram"] != row_b["ram"]:
        return True, "ram_mismatch"

    cond_a = row_a.get("condition")
    cond_b = row_b.get("condition")

    if cond_a and cond_b and cond_a != cond_b:
        return True, "condition_mismatch"

    if (cond_a in {"refurbished", "used"} and cond_b != cond_a) or (cond_b in {"refurbished", "used"} and cond_a != cond_b):
        return True, "condition_mismatch"

    return False, "no_conflict"


def is_strong_identity_match(row_a, row_b) -> tuple[bool, str]:
    if row_a.get("normalized_title") == row_b.get("normalized_title"):
        conflict, _ = has_hard_conflict(row_a, row_b)
        if not conflict:
            return True, "exact_normalized_match"

    if (
        row_a.get("category") == "phone"
        and row_b.get("category") == "phone"
        and row_a.get("brand") == row_b.get("brand")
        and row_a.get("model_family") == row_b.get("model_family")
        and row_a.get("model_number") == row_b.get("model_number")
        and row_a.get("storage") == row_b.get("storage")
        and row_a.get("tier_variant") == row_b.get("tier_variant")
        and row_a.get("condition") == row_b.get("condition")
        and row_a.get("color") == row_b.get("color")
    ):
        return True, "strong_phone_identity_match"

    if (
        row_a.get("brand") == row_b.get("brand")
        and row_a.get("model_family") == row_b.get("model_family")
        and row_a.get("model_number") == row_b.get("model_number")
        and row_a.get("tier_variant") == row_b.get("tier_variant")
        and row_a.get("display_variant") == row_b.get("display_variant")
        and row_a.get("connectivity") == row_b.get("connectivity")
        and row_a.get("ram") == row_b.get("ram")
        and row_a.get("storage") == row_b.get("storage")
        and row_a.get("screen_size") == row_b.get("screen_size")
        and row_a.get("condition") == row_b.get("condition")
        and row_a.get("color") == row_b.get("color")
    ):
        return True, "strong_model_identity_match"

    return False, ""


def compute_fuzzy_score(title_a: str, title_b: str) -> float:
    return (
        fuzz.token_sort_ratio(title_a, title_b)
        + fuzz.token_set_ratio(title_a, title_b)
        + fuzz.partial_ratio(title_a, title_b)
    ) / 300.0


def compute_attribute_agreement(row_a, row_b) -> float:
    fields = [
        "brand",
        "model_family",
        "model_number",
        "tier_variant",
        "display_variant",
        "connectivity",
        "storage",
        "ram",
        "screen_size",
        "condition",
        "color",
    ]

    matches = 0
    comparable = 0

    for field in fields:
        a = row_a.get(field)
        b = row_b.get(field)
        if a is not None and b is not None:
            comparable += 1
            if a == b:
                matches += 1

    return matches / comparable if comparable else 0.0


def compute_embedding_score(title_a: str, title_b: str, embedding_model=None) -> float:
    if embedding_model is None:
        return 0.0
    emb = embedding_model.encode([title_a, title_b], normalize_embeddings=True)
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])


def compute_match_score(fuzzy_score: float, embedding_score: float, attribute_score: float) -> float:
    """
    Weights chosen by reasoning about signal reliability on this dataset:
    - Fuzzy (0.50): strongest for mixed Hebrew/English titles
    - Attribute (0.35): high precision when present
    - Embedding (0.15): helps multilingual similarity but adds noise on variants

    Validated on labeled dataset; production would require tuning.
    """
    return 0.50 * fuzzy_score + 0.35 * attribute_score + 0.15 * embedding_score

def detect_risk_flags(row_a, row_b, final_score: float) -> list[str]:
    flags = []

    same_core = _same_core_identity(row_a, row_b)
    var_a = row_a.get("tier_variant")
    var_b = row_b.get("tier_variant")

    # Key fix:
    # Let LLM verify base vs special variant when only one side has explicit variant evidence
    if same_core:
        if (var_a in SPECIAL_TIER_VARIANTS and var_b is None) or (var_b in SPECIAL_TIER_VARIANTS and var_a is None):
            flags.append("base_vs_special_variant_missing_side")

    if final_score >= 0.90:
        if same_core and row_a.get("display_variant") != row_b.get("display_variant"):
            if row_a.get("display_variant") is None or row_b.get("display_variant") is None:
                flags.append("display_variant_partial_disagreement")

        if same_core and row_a.get("connectivity") != row_b.get("connectivity"):
            if row_a.get("connectivity") is None or row_b.get("connectivity") is None:
                flags.append("connectivity_partial_disagreement")

    if final_score >= 0.80 and same_core:
        if row_a.get("color") != row_b.get("color"):
            if row_a.get("color") is None or row_b.get("color") is None:
                flags.append("color_missing_side")

    return flags


def decide_match(row_a, row_b, final_score: float) -> tuple[str, str]:
    conflict, reason = has_hard_conflict(row_a, row_b)
    if conflict:
        return "reject", reason

    strong_match, strong_reason = is_strong_identity_match(row_a, row_b)
    if strong_match:
        return "merge", strong_reason

    risk_flags = detect_risk_flags(row_a, row_b, final_score)

    if final_score >= 0.985 and not risk_flags:
        return "merge", "very_high_score_no_conflict"

    if risk_flags:
        return "llm_verify", "|".join(risk_flags)

    if 0.78 <= final_score < 0.985:
        return "llm_review", "borderline_score_review"

    return "reject", "low_or_insufficient_confidence"


def run_matching(df: pd.DataFrame, embedding_model=None) -> pd.DataFrame:
    block_index = build_block_index(df)
    pairs = generate_candidate_pairs(block_index)

    id_map = {r["listing_id"]: r.to_dict() for _, r in df.iterrows()}
    results = []

    for a, b in sorted(pairs):
        row_a = id_map[a]
        row_b = id_map[b]

        title_a = row_a["normalized_title"]
        title_b = row_b["normalized_title"]

        conflict, conflict_reason = has_hard_conflict(row_a, row_b)
        if conflict:
            results.append({
                "listing_id_1": a,
                "listing_id_2": b,
                "title_1": row_a["title"],
                "title_2": row_b["title"],
                "normalized_title_1": title_a,
                "normalized_title_2": title_b,
                "fuzzy_score": 0.0,
                "embedding_score": 0.0,
                "attribute_score": 0.0,
                "final_score": 0.0,
                "decision": "reject",
                "reason": conflict_reason,
            })
            continue

        fuzzy_score = compute_fuzzy_score(title_a, title_b)
        attribute_score = compute_attribute_agreement(row_a, row_b)
        embedding_score = compute_embedding_score(title_a, title_b, embedding_model)
        final_score = compute_match_score(fuzzy_score, embedding_score, attribute_score)

        decision, reason = decide_match(row_a, row_b, final_score)

        results.append({
            "listing_id_1": a,
            "listing_id_2": b,
            "title_1": row_a["title"],
            "title_2": row_b["title"],
            "normalized_title_1": title_a,
            "normalized_title_2": title_b,
            "fuzzy_score": round(fuzzy_score, 4),
            "embedding_score": round(embedding_score, 4),
            "attribute_score": round(attribute_score, 4),
            "final_score": round(final_score, 4),
            "decision": decision,
            "reason": reason,
        })

    return pd.DataFrame(results)