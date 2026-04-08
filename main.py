import os
import time
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from attribute_extraction import extract_attributes
from matching import build_block_keys, run_matching, has_hard_conflict
from llm_layer import ask_llm_match
from postprocessing import create_grouped_products
from evaluation import run_evaluation
from analysis_layer import save_analysis


def read_products_csv(path: str) -> pd.DataFrame:
    encodings_to_try = [
        "utf-8",
        "utf-8-sig",
        "cp1255",
        "iso-8859-8",
        "cp1252",
        "latin1",
    ]

    last_error = None
    for enc in encodings_to_try:
        try:
            print(f"Trying to read CSV with encoding: {enc}")
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise ValueError(f"Could not decode CSV file: {path}. Last error: {last_error}")


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["title"] = df["title"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df[
        df["title"].notna() &
        df["price"].notna() &
        (df["title"].astype(str).str.strip() != "") &
        (df["title"].astype(str).str.strip().str.lower() != "nan") &
        (df["price"] > 0)
    ]

    return df.reset_index(drop=True)


def add_data_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["title"] = df["title"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()
    df["currency"] = df["currency"].astype(str).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["dq_missing_title"] = df["title"].isna() | (df["title"].astype(str).str.strip() == "")
    df["dq_missing_price"] = df["price"].isna()
    df["dq_nonpositive_price"] = df["price"].fillna(0) <= 0
    df["dq_suspicious_low_price"] = df["price"].fillna(0) < 200

    return df


def normalize_variant(value):
    if isinstance(value, str):
        value = value.strip().lower()
        return value if value else None
    return None


def is_dangerous_variant_mismatch(product_a: dict, product_b: dict) -> bool:
    va = normalize_variant(product_a.get("tier_variant"))
    vb = normalize_variant(product_b.get("tier_variant"))

    special_variants = {"ultra", "pro", "plus", "fe"}

    # explicit different variants
    if va is not None and vb is not None and va != vb:
        return True

    # base vs special
    if (va is None and vb in special_variants) or (vb is None and va in special_variants):
        return True

    return False


def main():
    load_dotenv()

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "openrouter/free")

    # 1. Load data
    df = read_products_csv("data/products.csv")
    df = clean_products(df)

    required_cols = {"listing_id", "title", "price", "source", "currency"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing_cols)}")

    # 2. Data quality flags
    df = add_data_quality_flags(df)
    df = df.replace(r"^\s*$", None, regex=True)

    # 3. Attribute extraction
    attr_df = df["title"].fillna("").apply(lambda x: pd.Series(extract_attributes(x)))
    df = pd.concat([df, attr_df], axis=1)
    df = df.replace({pd.NA: None})

    # 4. Keep only valid rows for matching
    matchable_df = df[
        (~df["dq_missing_title"]) &
        (~df["dq_missing_price"]) &
        (~df["dq_nonpositive_price"])
    ].copy()

    # 5. Build block keys
    matchable_df["block_keys"] = matchable_df.apply(build_block_keys, axis=1)

    # 6. Load embedding model
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # 7. Run base matching
    matches_df = run_matching(matchable_df, embedding_model=embedding_model)

    # 8. Prepare analytics columns
    for col in [
        "llm_confidence",
        "llm_raw_decision",
        "llm_prompt_version",
        "llm_route",
        "pipeline_version",
        "score_version",
        "review_bucket",
    ]:
        if col not in matches_df.columns:
            matches_df[col] = None

    # 9. LLM review / verify rows
    llm_rows_before = matches_df[
        matches_df["decision"].isin(["llm_review", "llm_verify"])
    ].copy()

    if openrouter_api_key:
        client = OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "zap-dedup-project",
            },
        )

        id_to_row = {row["listing_id"]: row.to_dict() for _, row in matchable_df.iterrows()}

        for idx, row in llm_rows_before.iterrows():
            product_a = id_to_row[row["listing_id_1"]]
            product_b = id_to_row[row["listing_id_2"]]

            matches_df.at[idx, "llm_route"] = row["decision"]
            matches_df.at[idx, "pipeline_version"] = "v2_routed_llm_verifier_guarded_yes_merge"
            matches_df.at[idx, "score_version"] = "fuzzy50_attr35_embed15"
            matches_df.at[idx, "review_bucket"] = "verify" if row["decision"] == "llm_verify" else "review"

            # Hard conflicts always win before LLM
            conflict, reason = has_hard_conflict(product_a, product_b)
            if conflict:
                matches_df.at[idx, "decision"] = "reject"
                matches_df.at[idx, "reason"] = reason
                continue

            llm_result = ask_llm_match(
                product_a=product_a,
                product_b=product_b,
                model=openrouter_model,
                client=client,
                evidence={
                    "route": row["decision"],
                    "fuzzy_score": row.get("fuzzy_score"),
                    "embedding_score": row.get("embedding_score"),
                    "attribute_score": row.get("attribute_score"),
                    "final_score": row.get("final_score"),
                    "rule_reason_before_llm": row.get("reason"),
                }
            )

            matches_df.at[idx, "llm_confidence"] = llm_result.get("confidence")
            matches_df.at[idx, "llm_raw_decision"] = llm_result.get("match")
            matches_df.at[idx, "llm_prompt_version"] = "v4_evidence_routed"

            rule_score = row["final_score"]
            llm_match = llm_result.get("match")
            llm_conf = float(llm_result.get("confidence", 0.3))

            # LLM can finalize, but never allow dangerous variant YES merges
            if llm_match == "yes" and llm_conf >= 0.5:
                if is_dangerous_variant_mismatch(product_a, product_b):
                    matches_df.at[idx, "decision"] = "reject"
                    matches_df.at[idx, "reason"] = "guardrail_variant_after_llm_yes"
                else:
                    matches_df.at[idx, "decision"] = "merge"
                    matches_df.at[idx, "reason"] = f"llm_merge: {llm_result.get('reason', 'no_reason')}"
            else:
                matches_df.at[idx, "decision"] = "reject"
                matches_df.at[idx, "reason"] = f"llm_reject: {llm_result.get('reason', 'no_reason')}"

            if llm_match == "no" and rule_score > 0.9:
                matches_df.at[idx, "reason"] += " | llm_rule_disagreement"

            time.sleep(1)

    else:
        mask = matches_df["decision"].isin(["llm_review", "llm_verify"])
        matches_df.loc[mask, "decision"] = "reject"
        matches_df.loc[mask, "reason"] = "llm_unavailable_conservative_reject"

    # 10. Console summaries
    llm_rows_after = matches_df[matches_df["reason"].astype(str).str.startswith("llm_")]

    print("\n=== LLM DECISION SUMMARY ===")
    print(f"LLM-reviewed pairs: {len(llm_rows_before)}")
    if not llm_rows_after.empty:
        print(llm_rows_after["decision"].value_counts())
    else:
        print("No LLM-reviewed rows finalized.")

    print("\n=== RULE vs LLM ===")
    print("Total pairs:", len(matches_df))
    print("LLM used:", len(llm_rows_before))
    print("Rule-only decisions:", len(matches_df) - len(llm_rows_before))

    llm_conf_series = pd.to_numeric(matches_df["llm_confidence"], errors="coerce").dropna()
    if not llm_conf_series.empty:
        print("\n=== LLM ANALYTICS ===")
        print("Avg LLM confidence:", round(float(llm_conf_series.mean()), 3))

        disagreements = matches_df[
            matches_df["reason"].astype(str).str.contains("llm_rule_disagreement", na=False)
        ]
        print("LLM vs Rule disagreements:", len(disagreements))

    # 11. Group output
    grouped_df = create_grouped_products(matchable_df, matches_df)

    # 12. Save outputs
    os.makedirs("outputs", exist_ok=True)
    matches_df.to_csv("outputs/match_decisions.csv", index=False, encoding="utf-8-sig")
    grouped_df.to_csv("outputs/grouped_products.csv", index=False, encoding="utf-8-sig")
    df.to_csv("outputs/products_with_attributes.csv", index=False, encoding="utf-8-sig")

    # 13. Evaluation
    run_evaluation(matches_df, grouped_df)
    print("\n=== PRODUCTION SCALE NOTES ===")
    print("Attribute extraction uses regex for known categories.")
    print("At Zap scale, this should be replaced with LLM or NER-based extraction.")
    print("Scoring weights were validated on this dataset; production requires tuning.")

    
    # 14. Analysis layer
    save_analysis(matches_df)

    print("\nDone.")
    print("Saved:")
    print("- outputs/products_with_attributes.csv")
    print("- outputs/match_decisions.csv")
    print("- outputs/grouped_products.csv")
    print("- outputs/llm_error_analysis.csv")
    print("- outputs/llm_usage_summary.csv")


if __name__ == "__main__":
    main()