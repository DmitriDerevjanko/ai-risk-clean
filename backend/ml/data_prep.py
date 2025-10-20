from __future__ import annotations
import pandas as pd


NEEDED_COLS = [
    "eventid", "iyear", "imonth", "iday",
    "region_txt", "country_txt",
    "nkill", "latitude", "longitude"
]


def load_gtd_csv(path: str) -> pd.DataFrame:
    """Load the GTD CSV and select required columns."""
    df = pd.read_csv(path, low_memory=False, encoding='latin1')
    required = ["iyear", "imonth", "iday", "region_txt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "eventid" not in df.columns:
        df["eventid"] = range(1, len(df) + 1)
    cols = [c for c in NEEDED_COLS if c in df.columns]
    return df[cols].copy()


def safe_date(row) -> pd.Timestamp:
    """Safely create a valid timestamp."""
    y = int(row.get("iyear", 1970))
    m = int(row.get("imonth", 1)) or 1
    d = int(row.get("iday", 1)) or 1
    m = min(max(m, 1), 12)
    d = min(max(d, 1), 28)
    return pd.Timestamp(year=y, month=m, day=d)


def to_monthly_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GTD data by region × month."""
    df = df.copy()

    # Filter invalid coordinates
    if "latitude" in df.columns and "longitude" in df.columns:
        df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)]

    # Create a date column
    df["date"] = df.apply(safe_date, axis=1)
    df["month"] = df["date"].values.astype("datetime64[M]")

    # Targets
    df["incidents_count"] = 1
    df["casualties"] = df.get("nkill", 0).fillna(0).clip(lower=0)

    grouped = (
        df.groupby(["region_txt", "month"], dropna=False)
        .agg(incidents_count=("incidents_count", "sum"),
             casualties=("casualties", "sum"))
        .reset_index()
        .rename(columns={"region_txt": "region"})
    )

    # Fill missing months
    def complete_months(grp: pd.DataFrame) -> pd.DataFrame:
        idx = pd.date_range(grp["month"].min(), grp["month"].max(), freq="MS")
        grp = grp.set_index("month").reindex(idx).rename_axis("month").reset_index()
        grp["region"] = grp["region"].ffill().bfill()
        grp["incidents_count"] = grp["incidents_count"].fillna(0)
        grp["casualties"] = grp["casualties"].fillna(0)
        return grp

    grouped = grouped.groupby("region", group_keys=False).apply(complete_months).reset_index(drop=True)
    return grouped
