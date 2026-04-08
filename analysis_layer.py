import os
import pandas as pd


def save_analysis(matches_df: pd.DataFrame) -> None:
    os.makedirs("outputs", exist_ok=True)

    # LLM usage summary
    llm_used = matches_df[matches_df["llm_route"].notna()].copy()
    usage_summary = pd.DataFrame([{
        "total_pairs": len(matches_df),
        "llm_used": len(llm_used),
        "llm_used_pct": round(len(llm_used) / len(matches_df) * 100, 2)
    }])

    usage_summary.to_csv(
        "outputs/llm_usage_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Detailed variant confusion cases (NOT just count)
    variant_confusion_pairs = matches_df[
        matches_df["reason"].astype(str).str.contains("variant", case=False, na=False) &
        (matches_df["decision"] == "reject")
    ][[
        "listing_id_1",
        "listing_id_2",
        "title_1",
        "title_2",
        "final_score",
        "llm_raw_decision",
        "llm_confidence",
        "reason"
    ]].copy()

    variant_confusion_pairs["error_type"] = "variant_confusion"

    # Summary row
    summary_row = pd.DataFrame([{
        "listing_id_1": "SUMMARY",
        "listing_id_2": f"total: {len(variant_confusion_pairs)}",
        "title_1": "",
        "title_2": "",
        "final_score": "",
        "llm_raw_decision": "",
        "llm_confidence": "",
        "reason": "variant_confusion cases — LLM correctly rejected base vs special variant pairs",
        "error_type": "variant_confusion"
    }])

    error_df = pd.concat([summary_row, variant_confusion_pairs], ignore_index=True)

    error_df.to_csv(
        "outputs/llm_error_analysis.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== LLM USAGE ===")
    print(f"Total pairs: {len(matches_df)}")
    print(f"LLM used: {len(llm_used)} ({round(len(llm_used) / len(matches_df) * 100, 2)}%)")

    if not llm_used.empty:
        print("\nLLM route breakdown:")
        print(
            llm_used.groupby("llm_route")
            .size()
            .reset_index(name="count")
            .to_string(index=False)
        )

    print("\n=== LLM ERROR ANALYSIS ===")
    if variant_confusion_pairs.empty:
        print("No variant_confusion cases found.")
    else:
        print(
            variant_confusion_pairs[["error_type", "reason"]]
            .value_counts()
            .head(5)
            .to_string()
        )