import pandas as pd
import json
import ast
from pathlib import Path


# ============================================================
# 1. DEFINE YOUR SELECTED IMAGES HERE (filenames only)
# ============================================================

# Example:
# SELECTED_IMAGES = [
#     "10x_qc1_qd2_23_00002.jpg",
#     "5x_sample_00123.png",
#     ...
# ]

SELECTED_IMAGES = [  "10x_qc1_qd2_23_00002.jpg"
                   , "10x_qc1_qd2_2_00002.jpg"
                   , "10x_qc1_qd2_3_00004.jpg"
                   , "agua2SDS_minOIL_qc16_qd1_form_00000.jpg"
                   , "agua2SDS_minOIL_qc2_qd1_const_00000.jpg"
                   , "agua2SDS_minOIL_qc4_qd1_form_00000.jpg"
                   , "formacao_00000.jpg"
                   , "oleo3_aguaup1_5x_formacao_00000.jpg"
                   , "oleo4_agua2_10x_formacao2_00000.jpg"
                   , "qc2_qd1_10x_formacao2_00000.jpg"
                   , "qc2_qd1_10x_formacao_00000.jpg"
                   , "qd2qc2_5x_1_00000.jpg"
                   , "qd2qc4_5x_1_00004.jpg"
                   , "qd2qc6_5x_partefinal_1_00001.jpg"
                   , "qd2qc7_5x_1_00000.jpg"]  # <-- 15 FILENAMES

TASK_IDS = [1,6,20,28,37,54,61,83,103,133,140,152,172,210,222]


# ============================================================
# 2. CSVs FOR EACH MODE
# ============================================================

CSV_FILES = {
    "manual_only": "annotations_manual.csv",
    "preannotations_only": "annotations_pre.csv",
    "preannotations_plus_sam2": "annotations_pre+sam2.csv"
}

COL_IMAGE = "image"
COL_LABEL = "label"
COL_SAM2 = "sam_mask"
COL_LEAD_TIME = "lead_time"


# ============================================================
# PARSERS
# ============================================================

def safe_parse_json(value):
    if value is None:
        return []
    s = str(value).strip()
    if not s or s == "[]":
        return []
    try:
        return json.loads(s)
    except:
        try:
            return ast.literal_eval(s)
        except:
            return []


def count_polygons(label_field):
    items = safe_parse_json(label_field)
    return len(items)


def count_sam_masks(mask_field):
    items = safe_parse_json(mask_field)
    return len(items)


# ============================================================
# PROCESS ONE CSV
# ============================================================

def process_single_csv(csv_path, mode_name):
    df = pd.read_csv(csv_path)

    # extract filename
    df["filename"] = df[COL_IMAGE].apply(lambda x: Path(str(x)).name)

    # filter by selected images
    df = df[df["filename"].isin(SELECTED_IMAGES)].copy()

    # bubble counts
    df["polygon_bubbles"] = df[COL_LABEL].apply(count_polygons)
    df["sam2_bubbles"] = df[COL_SAM2].apply(count_sam_masks)
    df["total_bubbles"] = df["polygon_bubbles"] + df["sam2_bubbles"]

    # time metrics
    df["sec_per_bubble"] = df.apply(
        lambda row: row[COL_LEAD_TIME] / row["total_bubbles"]
        if row["total_bubbles"] > 0 else None,
        axis=1,
    )

    df["bubbles_per_min"] = df.apply(
        lambda row: 60 * row["total_bubbles"] / row[COL_LEAD_TIME]
        if row[COL_LEAD_TIME] > 0 else None,
        axis=1,
    )

    df["mode"] = mode_name

    return df[
        [
            "filename",
            "mode",
            COL_LEAD_TIME,
            "polygon_bubbles",
            "sam2_bubbles",
            "total_bubbles",
            "sec_per_bubble",
            "bubbles_per_min",
        ]
    ]


# ============================================================
# MAIN
# ============================================================

def main():
    if len(SELECTED_IMAGES) == 0:
        print("ERROR: SELECTED_IMAGES list is empty. Fill it first!")
        return

    all_dfs = []

    for mode_name, csv_path in CSV_FILES.items():
        print(f"\nProcessing mode '{mode_name}' from {csv_path}...")
        df_mode = process_single_csv(csv_path, mode_name)
        all_dfs.append(df_mode)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv("annotation_efficiency_filtered_long.csv", index=False)
    print("\nSaved: annotation_efficiency_filtered_long.csv")

    # pivot to wide format
    wide = df_all.pivot_table(
        index="filename",
        columns="mode",
        values=[
            COL_LEAD_TIME,
            "total_bubbles",
            "sec_per_bubble",
            "bubbles_per_min",
        ],
        aggfunc="mean",
    )

    wide.columns = [f"{metric}_{mode}" for metric, mode in wide.columns]
    wide.reset_index(inplace=True)

    wide.to_csv("annotation_efficiency_filtered_wide.csv", index=False)
    print("Saved: annotation_efficiency_filtered_wide.csv")

    # summary
    summary = df_all.groupby("mode").agg(
        {
            COL_LEAD_TIME: ["mean", "median"],
            "total_bubbles": ["mean"],
            "sec_per_bubble": ["mean", "median"],
            "bubbles_per_min": ["mean", "median"],
        }
    )

    summary.to_csv("annotation_efficiency_filtered_summary_by_mode.csv")
    print("Saved: annotation_efficiency_filtered_summary_by_mode.csv")

    print("\nSUMMARY BY MODE:")
    print(summary)


if __name__ == "__main__":
    main()
