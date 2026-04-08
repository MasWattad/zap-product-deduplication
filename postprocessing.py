import pandas as pd


def group_has_internal_conflict(group_df: pd.DataFrame) -> tuple[bool, str]:
    conflict_fields = [
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
    ]

    for field in conflict_fields:
        values = {v for v in group_df[field].dropna().unique().tolist() if v is not None}
        if len(values) > 1:
            return True, f"group_{field}_conflict"

    return False, "no_conflict"


def build_groups(df_products: pd.DataFrame, df_matches: pd.DataFrame) -> dict:
    parent = {}
    for listing_id in df_products["listing_id"]:
        parent[listing_id] = listing_id

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    merges = df_matches[df_matches["decision"] == "merge"]

    for _, row in merges.iterrows():
        a = row["listing_id_1"]
        b = row["listing_id_2"]
        union(a, b)

    groups = {}
    for listing_id in df_products["listing_id"]:
        root = find(listing_id)
        if root not in groups:
            groups[root] = []
        groups[root].append(listing_id)

    return groups

def detect_suspicious_prices(group_df: pd.DataFrame) -> list[int]:
    prices = pd.to_numeric(group_df["price"], errors="coerce").dropna()

    if len(prices) < 2:
        return []

    median_price = float(prices.median())

    suspicious_ids = group_df.loc[
        pd.to_numeric(group_df["price"], errors="coerce") < (0.5 * median_price),
        "listing_id"
    ].tolist()

    return suspicious_ids


def choose_canonical_title(group_df: pd.DataFrame) -> str:
    group_df = group_df.copy()

    suspicious_ids = detect_suspicious_prices(group_df)
    if suspicious_ids:
        filtered_df = group_df[~group_df["listing_id"].isin(suspicious_ids)].copy()
        if not filtered_df.empty:
            group_df = filtered_df

    def completeness_score(row) -> int:
        return sum(
            1 for field in [
                "brand", "model_family", "model_number", "tier_variant",
                "display_variant", "connectivity", "storage", "ram",
                "screen_size", "condition", "color"
            ]
            if pd.notna(row.get(field))
        )

    def promo_penalty(title: str) -> int:
        title = str(title).lower()
        penalties = 0
        noisy_terms = ["מבצע", "sale", "חדש", "original", "יבואן", "official"]
        for term in noisy_terms:
            if term in title:
                penalties += 1
        return penalties

    group_df["title_length"] = group_df["title"].astype(str).str.len()
    group_df["completeness_score"] = group_df.apply(completeness_score, axis=1)
    group_df["promo_penalty"] = group_df["title"].apply(promo_penalty)

    best = group_df.sort_values(
        ["completeness_score", "promo_penalty", "title_length", "price"],
        ascending=[False, True, False, True]
    ).iloc[0]

    return best["title"]



def compute_group_min_price(group_df: pd.DataFrame) -> float:
    suspicious_ids = detect_suspicious_prices(group_df)

    filtered_df = group_df.copy()
    if suspicious_ids:
        tmp = filtered_df[~filtered_df["listing_id"].isin(suspicious_ids)].copy()
        if not tmp.empty:
            filtered_df = tmp

    prices = pd.to_numeric(filtered_df["price"], errors="coerce").dropna()
    if not prices.empty:
        return float(prices.min())

    fallback_prices = pd.to_numeric(group_df["price"], errors="coerce").dropna()
    if not fallback_prices.empty:
        return float(fallback_prices.min())

    return float("nan")
    

def collect_match_metadata(group_listing_ids: list, df_matches: pd.DataFrame) -> tuple[str, float]:
    accepted = df_matches[df_matches["decision"] == "merge"].copy()

    relevant = accepted[
        accepted["listing_id_1"].isin(group_listing_ids) &
        accepted["listing_id_2"].isin(group_listing_ids)
    ]

    if relevant.empty:
        return "singleton", 1.0

    weakest_row = relevant.sort_values("final_score", ascending=True).iloc[0]
    confidence = float(weakest_row["final_score"])

    if isinstance(weakest_row.get("reason"), str) and "llm" in weakest_row["reason"]:
        return "llm", confidence

    return "rules_scoring", confidence


def create_grouped_products(df_products: pd.DataFrame, df_matches: pd.DataFrame) -> pd.DataFrame:
    groups = build_groups(df_products, df_matches)

    rows = []
    for idx, (_, listing_ids) in enumerate(groups.items(), start=1):
        group_df = df_products[df_products["listing_id"].isin(listing_ids)].copy()

        canonical_title = choose_canonical_title(group_df)
        min_price = compute_group_min_price(group_df)
        sources = sorted(group_df["source"].dropna().unique().tolist())
        match_source, confidence = collect_match_metadata(listing_ids, df_matches)
        has_conflict, conflict_reason = group_has_internal_conflict(group_df)

        suspicious_price_listing_ids = group_df.loc[
            group_df.get(
                "dq_suspicious_low_price",
                pd.Series(False, index=group_df.index)
            ).fillna(False),
            "listing_id"
        ].tolist()

        rows.append({
            "group_id": idx,
            "canonical_title": canonical_title,
            "listing_ids": listing_ids,
            "titles": group_df["title"].tolist(),
            "sources": sources,
            "min_price": min_price,
            "currency": group_df["currency"].iloc[0],
            "match_source": match_source,
            "confidence": round(confidence, 4),
            "group_conflict": has_conflict,
            "group_conflict_reason": conflict_reason,
            "suspicious_price_listing_ids": suspicious_price_listing_ids,
        })

    return pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)