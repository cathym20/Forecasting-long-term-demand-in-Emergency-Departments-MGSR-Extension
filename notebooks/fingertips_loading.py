"""Load ICB-level population-health features from OHID Fingertips for ED demand modelling.

Data is licensed under Open Government Licence v3.0
(cite: Office for Health Improvement & Disparities, Public Health Profiles, fingertips.phe.org.uk).
"""

import re
import pandas as pd
import fingertips_py as ftp


INDICATORS = [
    "Deprivation score (IMD 2019)",
    "COPD: QOF prevalence",
    "Diabetes: QOF prevalence",
    "Hypertension: QOF prevalence",
    "Asthma: QOF prevalence",
    "Depression: QOF prevalence",
    "Obesity: QOF prevalence",
    "CHD: QOF prevalence",
    "Heart Failure: QOF prevalence",
]

YEAR_RANGE = [2022, 2023, 2024]


def resolve_ids(names):
    """Match indicator names to Fingertips IDs. Fingertips search is picky about exact
    strings, so use a case-insensitive contains match and pick the shortest name that hits."""
    meta = ftp.get_metadata_for_all_indicators_from_csv()
    idcol = next(c for c in meta.columns if c.lower().strip() in ("indicator id", "indicatorid"))
    nmcol = next(c for c in meta.columns if c.lower().strip() == "indicator")
    ids = {}
    for n in names:
        hits = meta[meta[nmcol].str.contains(n, case=False, na=False, regex=False)]
        if len(hits):
            row = hits.loc[hits[nmcol].str.len().idxmin()]
            ids[int(row[idcol])] = row[nmcol]
    return list(ids)


def build_ons_to_nhs(population_file, ae_file):
    """Fingertips uses ONS ICB codes (E54...); the HES A&E data uses three-char NHS codes.
    Both files list ICBs by name so we join through a normalised name."""
    sheets = pd.ExcelFile(population_file).sheet_names
    mid = next((s for s in sheets if "Mid" in s), sheets[0])
    raw = pd.read_excel(population_file, sheet_name=mid, header=None, skiprows=4)
    pop = raw[[2, 3]].drop_duplicates().dropna()
    pop.columns = ["icb_ons", "pop_name"]
    pop["key"] = pop["pop_name"].str.lower().str.replace("integrated care board", "").str.replace("nhs", "").str.strip()

    ae = pd.read_excel(ae_file, engine="xlrd", sheet_name="System Level Data", header=None)
    codes = ae.iloc[17:, 1].astype(str).str.strip()
    names = ae.iloc[17:, 2].astype(str).str.strip()
    ae_df = pd.DataFrame({"icb_nhs": codes, "ae_name": names})
    ae_df = ae_df[ae_df["icb_nhs"].str.len() == 3].drop_duplicates("icb_nhs")
    ae_df["key"] = ae_df["ae_name"].str.lower().str.replace("integrated care board", "").str.replace("nhs", "").str.strip()

    merged = pop.merge(ae_df, on="key", how="inner")
    return dict(zip(merged["icb_ons"], merged["icb_nhs"]))


def build_sub_to_icb_ons(population_file):
    sheets = pd.ExcelFile(population_file).sheet_names
    mid = next((s for s in sheets if "Mid" in s), sheets[0])
    raw = pd.read_excel(population_file, sheet_name=mid, header=None, skiprows=4)
    return dict(zip(raw[0].astype(str).str.strip(), raw[2].astype(str).str.strip()))


def parse_year(time_str):
    """Fingertips reports periods either as a single year (2022) or a financial-year range
    (2022/23). Return the ending year in the latter case so 2022/23 lines up with 2023."""
    m = re.search(r"(\d{4})", str(time_str))
    if not m:
        return None
    y = int(m.group(1))
    if "/" in str(time_str):
        return y + 1
    return y


def filter_age(df, age_col):
    """Pick the widest available age band per indicator, defaulting to 'All ages'."""
    ages = df[age_col].astype(str).unique()
    for pref in ["All ages", "18+", "17+", "16+", "15+",
                 "18+ yrs", "17+ yrs", "16+ yrs", "15+ yrs", "15-74 yrs"]:
        if pref in ages:
            return df[df[age_col].astype(str) == pref]
    return df


def process_level(raw_df, level, sub_to_icb_ons, year_range):
    """Turn a long-format Fingertips response into a wide-format frame keyed by
    (icb_ons, year). Sub-ICB-level data is aggregated up to ICB by taking the mean."""
    if raw_df is None or len(raw_df) == 0:
        return None

    cols = {c.lower().strip(): c for c in raw_df.columns}
    A, I, V, T = cols["area code"], cols["indicator name"], cols["value"], cols["time period"]

    df = raw_df.copy()

    if "sex" in cols:
        sex_col = cols["sex"]
        if "Persons" in df[sex_col].unique():
            df = df[df[sex_col] == "Persons"]

    if "age" in cols:
        age_col = cols["age"]
        df[age_col] = df[age_col].astype(str)
        per_indicator = []
        for ind_name, grp in df.groupby(I):
            per_indicator.append(filter_age(grp, age_col))
        df = pd.concat(per_indicator, ignore_index=True)

    if level == "icb":
        df = df[df[A].astype(str).str.startswith("E54")]
    else:
        df = df[df[A].astype(str).str.startswith("E38")]
        df["icb_ons"] = df[A].map(sub_to_icb_ons)
        df = df.dropna(subset=["icb_ons"])

    df["year"] = df[T].apply(parse_year)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # separate annual QOF indicators from static ones like IMD 2019
    static = df[~df[T].astype(str).str.contains("/")]
    annual = df[df[T].astype(str).str.contains("/")]

    frames = []

    if len(annual):
        annual = annual[annual["year"].isin(year_range)]
        if level == "sub":
            annual = annual.groupby(["icb_ons", I, "year"])[V].mean().reset_index()
            key_col = "icb_ons"
        else:
            key_col = A
        for yr, ygrp in annual.groupby("year"):
            wide = ygrp.pivot_table(index=key_col, columns=I, values=V, aggfunc="mean")
            wide = wide.reset_index().rename(columns={key_col: "icb_ons"})
            wide["year"] = yr
            frames.append(wide)

    if len(static):
        if level == "sub":
            static = static.groupby(["icb_ons", I])[V].mean().reset_index()
            key_col = "icb_ons"
        else:
            key_col = A
        wide_static = static.pivot_table(index=key_col, columns=I, values=V, aggfunc="mean")
        wide_static = wide_static.reset_index().rename(columns={key_col: "icb_ons"})
        for yr in year_range:
            w = wide_static.copy()
            w["year"] = yr
            frames.append(w)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(["icb_ons", "year"]).first().reset_index()


def load_fingertips_icb(population_file, ae_file, out_path="../new_data/fingertips_icb_features.csv"):
    ids = resolve_ids(INDICATORS)
    ons_to_nhs = build_ons_to_nhs(population_file, ae_file)
    sub_to_icb_ons = build_sub_to_icb_ons(population_file)

    # Not every indicator is published at both levels, so fetch both and use Sub-ICB
    # aggregated up as a fallback for anything the ICB level is missing.
    icb_raw = ftp.get_data_by_indicator_ids(ids, 221)
    sub_raw = ftp.get_data_by_indicator_ids(ids, 66)

    icb_wide = process_level(icb_raw, "icb", sub_to_icb_ons, YEAR_RANGE)
    sub_wide = process_level(sub_raw, "sub", sub_to_icb_ons, YEAR_RANGE)

    if icb_wide is not None and sub_wide is not None:
        sub_only_cols = [c for c in sub_wide.columns
                         if c not in icb_wide.columns and c not in ("icb_ons", "year")]
        if sub_only_cols:
            combined = icb_wide.merge(
                sub_wide[["icb_ons", "year"] + sub_only_cols],
                on=["icb_ons", "year"], how="outer")
        else:
            combined = icb_wide
    elif icb_wide is not None:
        combined = icb_wide
    elif sub_wide is not None:
        combined = sub_wide
    else:
        raise RuntimeError("No data returned from either level")

    combined.insert(0, "icb_code", combined["icb_ons"].map(ons_to_nhs))
    combined = combined.dropna(subset=["icb_code"]).drop(columns=["icb_ons"])

    # normalise column names: fx_ prefix, alphanumeric only, capped at 50 chars
    combined.columns = [c if c in ("icb_code", "year") else
                        "fx_" + "".join(ch if ch.isalnum() else "_" for ch in c.lower()).strip("_")[:50]
                        for c in combined.columns]

    combined = combined.sort_values(["icb_code", "year"]).reset_index(drop=True)
    combined.to_csv(out_path, index=False)
    print(f"Saved {out_path}: {len(combined)} rows, {combined['icb_code'].nunique()} ICBs")
    return combined


if __name__ == "__main__":
    load_fingertips_icb(
        population_file="../data/population/sapeicb20222024.xlsx",
        ae_file="../data/ae/May-2023-AE-revised-130624-by-provider-jn7Oy.xls",
    )
