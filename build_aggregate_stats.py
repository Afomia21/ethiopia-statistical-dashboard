from pathlib import Path

import pandas as pd

DATA_DIR = Path("data set") / "ETH_2021_ESPS-W5_v02_M_CSV"
OUT_DIR = Path("data set") / "preprocessed"
OUT_FILE = OUT_DIR / "aggregate_stats.csv"

REGION_COL = "saq01"
AREA_COL = "saq14"


def clean_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"^\d+\.\s*", "", regex=True).str.strip()


def load_consumption() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "cons_agg_w5.csv")
    df[REGION_COL] = clean_code(df[REGION_COL])
    df[AREA_COL] = clean_code(df[AREA_COL])
    return df


def load_demographics() -> pd.DataFrame:
    cover = pd.read_csv(DATA_DIR / "sect_cover_hh_w5.csv", dtype=str)
    cover[REGION_COL] = clean_code(cover[REGION_COL])
    cover[AREA_COL] = clean_code(cover[AREA_COL])
    cover = cover[["household_id", REGION_COL, AREA_COL]]

    roster = pd.read_csv(DATA_DIR / "sect1_hh_w5.csv", dtype=str)
    roster["age"] = pd.to_numeric(clean_code(roster["s1q03a"]), errors="coerce")
    roster["sex"] = clean_code(roster["s1q02"])
    roster = roster[["household_id", "age", "sex"]]

    merged = roster.merge(cover, on="household_id", how="left")
    return merged


def load_education() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sect2_hh_w5.csv", dtype=str)
    df[REGION_COL] = clean_code(df[REGION_COL])
    df["can_read_write"] = clean_code(df["s2q03"])
    df["ever_attended_school"] = clean_code(df["s2q04"])
    return df


def load_health() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sect3_hh_w5.csv", dtype=str)
    df[REGION_COL] = clean_code(df[REGION_COL])
    df["had_illness_4wk"] = clean_code(df["s3q05"])
    return df


def stats_rows(group_cols, df, value_col, agg, label_template, indicator_name):
    rows = []
    grouped = df.groupby(group_cols)[value_col].agg(agg).reset_index()
    for _, r in grouped.iterrows():
        group_desc = ", ".join(str(r[c]) for c in group_cols)
        value = r[value_col]
        sentence = label_template.format(group=group_desc, value=value)
        rows.append({
            "group": group_desc,
            "indicator": indicator_name,
            "value": value,
            "text": sentence,
        })
    return rows


def build_all_stats() -> pd.DataFrame:
    all_rows = []

    print("Loading consumption data...")
    cons = load_consumption()

    print("Computing consumption stats by region...")
    all_rows += stats_rows(
        [REGION_COL], cons, "nom_totcons_aeq", "mean",
        "In {group}, the average total annual consumption per adult equivalent was {value:.0f} birr (ESPS-5, 2021/22).",
        "avg_total_consumption_per_adult_equiv"
    )
    all_rows += stats_rows(
        [REGION_COL, AREA_COL], cons, "nom_totcons_aeq", "mean",
        "In {group}, the average total annual consumption per adult equivalent was {value:.0f} birr (ESPS-5, 2021/22).",
        "avg_total_consumption_per_adult_equiv_by_area"
    )
    all_rows += stats_rows(
        [REGION_COL], cons, "hh_size", "mean",
        "In {group}, the average household size was {value:.1f} people (ESPS-5, 2021/22).",
        "avg_household_size"
    )

    print("Loading demographic data...")
    demo = load_demographics()

    print("Computing demographic stats by region...")
    all_rows += stats_rows(
        [REGION_COL], demo, "age", "mean",
        "In {group}, the average age of household members was {value:.1f} years (ESPS-5, 2021/22).",
        "avg_age"
    )

    demo["is_female"] = (demo["sex"].str.upper() == "FEMALE").astype(float) * 100
    all_rows += stats_rows(
        [REGION_COL], demo, "is_female", "mean",
        "In {group}, {value:.1f}% of surveyed household members were female (ESPS-5, 2021/22).",
        "pct_female"
    )

    print("Loading education data...")
    edu = load_education()
    edu["is_literate"] = (edu["can_read_write"].str.upper() == "YES").astype(float) * 100
    all_rows += stats_rows(
        [REGION_COL], edu, "is_literate", "mean",
        "In {group}, an estimated {value:.1f}% of surveyed individuals can read and write (ESPS-5, 2021/22).",
        "pct_literate"
    )
    edu["attended_school"] = (edu["ever_attended_school"].str.upper() == "YES").astype(float) * 100
    all_rows += stats_rows(
        [REGION_COL], edu, "attended_school", "mean",
        "In {group}, an estimated {value:.1f}% of surveyed individuals have ever attended school (ESPS-5, 2021/22).",
        "pct_ever_attended_school"
    )

    print("Loading health data...")
    health = load_health()
    health["ill_recent"] = (health["had_illness_4wk"].str.upper() == "YES").astype(float) * 100
    all_rows += stats_rows(
        [REGION_COL], health, "ill_recent", "mean",
        "In {group}, an estimated {value:.1f}% of surveyed individuals reported an illness or injury in the last 4 weeks (ESPS-5, 2021/22).",
        "pct_illness_4wk"
    )

    return pd.DataFrame(all_rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_df = build_all_stats()
    stats_df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(stats_df)} aggregate statistics to {OUT_FILE}")
    print("\nSample:")
    for t in stats_df["text"].head(8):
        print(" -", t)


if __name__ == "__main__":
    main()