import pandas as pd


def evaluate_against_labels(matches_df: pd.DataFrame, labels_path: str = "data/gold_labels.csv") -> None:
    gold_df = pd.read_csv(labels_path)

    gold_df["key"] = gold_df.apply(
        lambda r: tuple(sorted((int(r["listing_id_1"]), int(r["listing_id_2"])))),
        axis=1
    )

    pred_df = matches_df.copy()
    pred_df["key"] = pred_df.apply(
        lambda r: tuple(sorted((int(r["listing_id_1"]), int(r["listing_id_2"])))),
        axis=1
    )

    pred_df["pred_label"] = (pred_df["decision"] == "merge").astype(int)

    merged = gold_df.merge(
        pred_df[["key", "pred_label", "listing_id_1", "listing_id_2", "title_1", "title_2", "final_score", "reason"]],
        on="key",
        how="left"
    )

    merged["pred_label"] = merged["pred_label"].fillna(0).astype(int)

    tp = int(((merged["label"] == 1) & (merged["pred_label"] == 1)).sum())
    fp = int(((merged["label"] == 0) & (merged["pred_label"] == 1)).sum())
    tn = int(((merged["label"] == 0) & (merged["pred_label"] == 0)).sum())
    fn = int(((merged["label"] == 1) & (merged["pred_label"] == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(merged) if len(merged) else 0.0
    false_merge_rate = fp / (tp + fp) if (tp + fp) else 0.0

    print("\n=== LABELED EVALUATION ===")
    print(f"Evaluated labeled pairs: {len(merged)}")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"False merge rate: {false_merge_rate:.3f}")

    print("\nNote: gold labels were manually created by the pipeline author against")
    print("the same dataset. These metrics validate rule correctness on known cases,")
    print("not generalization to unseen data. A held-out blind evaluation would be")
    print("needed to confirm these numbers on real Zap inventory.")

    false_positives = merged[(merged["label"] == 0) & (merged["pred_label"] == 1)]
    false_negatives = merged[(merged["label"] == 1) & (merged["pred_label"] == 0)]

    print("\n--- False Positives (bad merges) ---")
    print("None" if false_positives.empty else false_positives.to_string(index=False))

    print("\n--- False Negatives (missed matches) ---")
    print("None" if false_negatives.empty else false_negatives.to_string(index=False))


def run_evaluation(matches_df: pd.DataFrame, grouped_df: pd.DataFrame) -> None:
    print("\n=== MATCH DECISION SUMMARY ===")
    print(f"Total candidate pairs: {len(matches_df)}")
    print(matches_df["decision"].value_counts())

    print("\n=== GROUP SUMMARY ===")
    print(f"Total product groups: {len(grouped_df)}")

    evaluate_against_labels(matches_df)