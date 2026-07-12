import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data set") / "ETH_2021_ESPS-W5_v02_M_CSV"
PREPROCESS_DIR = Path("data set") / "preprocessed"
CODEBOOK_PATH = Path("codebook.json")
MERGED_FILE = PREPROCESS_DIR / "merged_dataset.csv"

with open(CODEBOOK_PATH, encoding="utf-8") as f:
    CODEBOOK = json.load(f)

SKIP_COLS = {
    "household_id", "individual_id", "ea_id", "holder_id", "parcel_id",
    "field_id", "crop_id", "enterprise_id", "asset_cd", "item_cd", "item_cd_cf",
    "interview__key", "interview__id", "pw_w5", "saq09", "saq11", "saq13",
    "saq17", "saq18", "saq19__latitude", "saq19__longitude", "saq19__accuracy",
    "saq19__altitude", "saq19__timestamp", "saq21",
}

REGION_COL = "saq01"
URBAN_COL = "saq14"
NA_VALUES = {"", "na", "n/a", "none", "nan"}


def normalize_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9a-z_]+", "", name)
    return name

def make_unique(columns) -> list:
    seen = {}
    result = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
    return result


def get_file_labels(csv_name: str) -> dict:
    stem = Path(csv_name).stem
    for key in (f"{stem}.dta", stem):
        if key in CODEBOOK:
            return CODEBOOK[key]
    return {}


def clean_value(value) -> str:
    value = str(value).strip()
    value = re.sub(r"^\d+\.\s*", "", value)
    return value


def clean_label(label: str) -> str:
    label = str(label).strip()
    label = re.sub(r"^\d+[a-z]?\.\s*", "", label)
    label = label.replace("[NAME]", "this person")
    label = label.rstrip("?")
    return label


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = make_unique([normalize_column_name(c) for c in df.columns])
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({v: pd.NA for v in NA_VALUES})
    return df


def row_to_text(row: pd.Series, labels: dict) -> str:
    parts = []
    for col, value in row.items():
        if col in SKIP_COLS or col in (REGION_COL, URBAN_COL):
            continue
        if pd.isna(value):
            continue
        text_value = clean_value(value)
        if text_value == "":
            continue
        label = clean_label(labels.get(col, col))
        parts.append(f"{label}: {text_value}")
    return ". ".join(parts)


def build_context_index(data_dir: Path) -> dict:
    context = {}

    cover_path = data_dir / "sect_cover_hh_w5.csv"
    roster_path = data_dir / "sect1_hh_w5.csv"

    if cover_path.exists():
        cover = pd.read_csv(cover_path, dtype=str, low_memory=False, na_filter=False)
        cover = clean_dataframe(cover)
        for _, r in cover.iterrows():
            hh = r.get("household_id")
            if not hh:
                continue
            context.setdefault(hh, {})
            context[hh]["region"] = clean_value(r.get(REGION_COL, ""))
            context[hh]["area"] = clean_value(r.get(URBAN_COL, ""))

    if roster_path.exists():
        roster = pd.read_csv(roster_path, dtype=str, low_memory=False, na_filter=False)
        roster = clean_dataframe(roster)
        for _, r in roster.iterrows():
            hh = r.get("household_id")
            ind = r.get("individual_id")
            if not hh or not ind:
                continue
            household_ctx = context.get(hh, {})
            context[(hh, ind)] = {
                "region": household_ctx.get("region", ""),
                "area": household_ctx.get("area", ""),
                "age": clean_value(r.get("s1q03a", "")),
                "sex": clean_value(r.get("s1q02", "")),
                "relationship_to_head": clean_value(r.get("s1q01", "")),
            }

    return context


def describe_context(hh, ind, context: dict) -> str:
    person = context.get((hh, ind)) if ind else None
    household = context.get(hh, {})
    details = []
    if person:
        if person.get("age"):
            details.append(f"age {person['age']}")
        if person.get("sex"):
            details.append(person["sex"].lower())
        if person.get("relationship_to_head"):
            details.append(f"({person['relationship_to_head']} of household)")

    region = (person or household).get("region")
    area = (person or household).get("area")

    header = f"Household {hh}"
    if ind:
        header += f", member {ind}"
    if details:
        header += " (" + ", ".join(details) + ")"
    if region:
        header += f" — {region}"
    if area:
        header += f", {area} area"
    return header + "."


def preprocess_files(data_dir: Path, preprocess_dir: Path) -> pd.DataFrame:
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    print("Building household/individual context index...")
    context = build_context_index(data_dir)

    rows = []
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    for csv_path in csv_files:
        print(f"Processing {csv_path.name}")
        labels = get_file_labels(csv_path.name)
        if not labels:
            print(f"  WARNING: no codebook labels found for {csv_path.name} — will use raw column codes")

        df = pd.read_csv(csv_path, dtype=str, low_memory=False, na_filter=False)
        df = clean_dataframe(df)

        cleaned_path = preprocess_dir / f"cleaned_{csv_path.name}"
        df.to_csv(cleaned_path, index=False)

        has_individual = "individual_id" in df.columns

        for row_index, row in df.iterrows():
            hh = row.get("household_id")
            ind = row.get("individual_id") if has_individual else None

            facts = row_to_text(row, labels)
            if not facts:
                continue

            header = describe_context(hh, ind, context) if hh else ""
            full_text = f"{header} {facts}".strip()

            rows.append({
                "text": full_text,
                "household_id": hh,
                "individual_id": ind,
                "source_file": csv_path.name,
                "row_index": row_index,
            })

    return pd.DataFrame(rows)


def main() -> None:
    merged_df = preprocess_files(DATA_DIR, PREPROCESS_DIR)
    print(f"Saving merged dataset to {MERGED_FILE}")
    merged_df.to_csv(MERGED_FILE, index=False)
    print(f"Merged dataset saved with {len(merged_df)} rows")
    print("\nSample rows:")
    print(merged_df["text"].head(3).to_string())


if __name__ == "__main__":
    main()