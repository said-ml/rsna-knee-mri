from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/workspace")

GOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "report_001_gold.csv"
)

PRED_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "report_001_v1_predictions.csv"
)

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def main():

    gold = pd.read_csv(GOLD_PATH)
    pred = pd.read_csv(PRED_PATH)

    df = gold[
        ["StudyInstanceUID"] + TARGETS
    ].merge(
        pred,
        on="StudyInstanceUID",
        suffixes=("_gold", "_pred"),
    )

    summary = []

    for target in TARGETS:

        gold_col = f"{target}_gold"
        pred_col = f"{target}_pred"

        evaluated = df[pred_col].notna()

        n_evaluated = int(evaluated.sum())

        if n_evaluated == 0:
            continue

        y_true = df.loc[evaluated, gold_col]
        y_pred = df.loc[evaluated, pred_col]

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        accuracy = (tp + tn) / n_evaluated

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else float("nan")
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else float("nan")
        )

        disagreement = int((y_true != y_pred).sum())

        summary.append(
            {
                "Target": target,
                "Coverage": n_evaluated / len(df),
                "N": n_evaluated,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "Disagreement": disagreement,
            }
        )

    summary_df = pd.DataFrame(summary)

    print("\n" + "=" * 100)
    print("REPORT-001 v1 — GOLD AGREEMENT")
    print("=" * 100)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # --------------------------------------------------------
    # Detailed disagreements
    # --------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("DETAILED DISAGREEMENTS")
    print("=" * 100)

    disagreement_rows = []

    for _, row in df.iterrows():

        for target in TARGETS:

            gold_value = row[f"{target}_gold"]
            pred_value = row[f"{target}_pred"]

            if pd.isna(pred_value):
                continue

            if gold_value != pred_value:

                disagreement_rows.append(
                    {
                        "StudyInstanceUID":
                            row["StudyInstanceUID"],
                        "Target":
                            target,
                        "Gold":
                            int(gold_value),
                        "Pred":
                            int(pred_value),
                        "Evidence":
                            row.get(
                                f"{target}__evidence",
                                "",
                            ),
                        "Language":
                            row.get(
                                "language_hint",
                                "",
                            ),
                        "Report":
                            row.get("Report", ""),
                    }
                )

    disagreements = pd.DataFrame(disagreement_rows)

    print(
        f"\nTotal report-vs-gold disagreements: "
        f"{len(disagreements)}"
    )

    if len(disagreements) > 0:

        for _, row in disagreements.iterrows():

            print("\n" + "-" * 100)
            print(
                f"Target:   {row['Target']}\n"
                f"Gold:     {row['Gold']}\n"
                f"Pred:     {row['Pred']}\n"
                f"Language: {row['Language']}\n"
                f"Evidence: {row['Evidence']}\n"
                f"Report:   {row['Report']}"
            )

    # --------------------------------------------------------
    # Save machine-readable audit
    # --------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "data"
        / "reports"
        / "report_001_v1_audit.csv"
    )

    disagreements.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved disagreement audit: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()