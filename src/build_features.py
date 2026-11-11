"""Join the parsed A&E table with AQHI, HKO daily weather and public holidays into an hourly modelling table.

Usage: python src/build_features.py [--ae data/processed/ae_snapshots.csv] [--aux data/raw/aux] [--out data/processed/hourly_features.csv]

Each row = one hospital x one hour. Columns:
  hospital, hour_ts, fmt, n_snap,
  targets : t45_min, t45_over2h, t3p50_min (new format only)
  calendar: dow, hr, is_weekend, is_holiday, month
  air     : aqhi (nearest EPD station, same hour)
  weather : rf_mm (daily total rainfall), rh (daily mean RH), temp_mean, temp_max, temp_min (HKO station, same day)
  lags    : t45_lag1 (previous hour, same hospital), t45_lag24 (same hour yesterday)
"""
import argparse, glob, io, json, os
import numpy as np
import pandas as pd

# Nearest EPD AQHI station for each public A&E department (by district).
HOSP_TO_STATION = {
    "Alice Ho Miu Ling Nethersole Hospital": "Tai Po",
    "Caritas Medical Centre": "Sham Shui Po",
    "Kwong Wah Hospital": "Mong Kok",
    "North District Hospital": "North",
    "North Lantau Hospital": "Tung Chung",
    "Pamela Youde Nethersole Eastern Hospital": "Eastern",
    "Pok Oi Hospital": "Yuen Long",
    "Prince of Wales Hospital": "Sha Tin",
    "Princess Margaret Hospital": "Kwai Chung",
    "Queen Elizabeth Hospital": "Mong Kok",
    "Queen Mary Hospital": "Central/Western",
    "Ruttonjee Hospital": "Causeway Bay",
    "St John Hospital": "Southern",
    "Tin Shui Wai Hospital": "Yuen Long",
    "Tseung Kwan O Hospital": "Tseung Kwan O",
    "Tuen Mun Hospital": "Tuen Mun",
    "United Christian Hospital": "Kwun Tong",
    "Yan Chai Hospital": "Tsuen Wan",
}


def load_aqhi(aux):
    """EPD monthly CSVs -> long table (ts, station, aqhi). Hour 24 = 00:00 next day. '*' marks provisional values."""
    frames = []
    for p in sorted(glob.glob(os.path.join(aux, "aqhi", "*_Eng.csv"))):
        lines = open(p, encoding="utf-8-sig").read().splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith("Date,Hour"))
        header = [h.strip() for h in lines[start].rstrip(",").split(",")]
        body = [l for l in lines[start + 1:] if l and not l.split(",")[1].strip().startswith("Daily")]
        df = pd.read_csv(io.StringIO("\n".join(body)), header=None, dtype=str)
        df = df.iloc[:, : len(header)]
        df.columns = header
        df["Date"] = df["Date"].replace("", np.nan).ffill()
        df = df.dropna(subset=["Date"])
        long = df.melt(id_vars=["Date", "Hour"], var_name="station", value_name="aqhi")
        long["aqhi"] = pd.to_numeric(long["aqhi"].astype(str).str.replace("*", "", regex=False).str.replace("+", "", regex=False), errors="coerce")
        long["ts"] = pd.to_datetime(long["Date"]) + pd.to_timedelta(long["Hour"].astype(int), unit="h")
        frames.append(long[["ts", "station", "aqhi"]])
    return pd.concat(frames, ignore_index=True)


def load_hko(aux):
    out = None
    for ele, col in (("RF", "rf_mm"), ("RH", "rh"), ("TEMP", "temp_mean"), ("MAXT", "temp_max"), ("MINT", "temp_min")):
        p = os.path.join(aux, "hko", f"daily_HKO_{ele}_ALL.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, skiprows=2, encoding="utf-8-sig", dtype=str, on_bad_lines="skip")
        df = df.iloc[:, :4]
        df.columns = ["y", "m", "d", col]
        df = df[df["y"].str.fullmatch(r"\d{4}", na=False)]
        df["date"] = pd.to_datetime(dict(year=df.y.astype(int), month=df.m.astype(int), day=df.d.astype(int)), errors="coerce")
        df[col] = pd.to_numeric(df[col].replace({"Trace": "0", "***": np.nan}), errors="coerce")
        df = df.dropna(subset=["date"])[["date", col]]
        out = df if out is None else out.merge(df, on="date", how="outer")
    return out


# The 1823 iCal feed only covers 2025 onwards, so 2023-2024 general holidays were silently 0 before 2026-10-30.
# Transcribed from gov.hk/en/about/abouthk/holiday/{2023,2024}.htm (17 days each, checked 2026-10-30).
HOLIDAYS_2023_2024 = [
    "2023-01-02", "2023-01-23", "2023-01-24", "2023-01-25", "2023-04-05", "2023-04-07", "2023-04-08",
    "2023-04-10", "2023-05-01", "2023-05-26", "2023-06-22", "2023-07-01", "2023-09-30", "2023-10-02",
    "2023-10-23", "2023-12-25", "2023-12-26",
    "2024-01-01", "2024-02-10", "2024-02-12", "2024-02-13", "2024-03-29", "2024-03-30", "2024-04-01",
    "2024-04-04", "2024-05-01", "2024-05-15", "2024-06-10", "2024-07-01", "2024-09-18", "2024-10-01",
    "2024-10-11", "2024-12-25", "2024-12-26",
]


def load_holidays(aux):
    h = json.load(open(os.path.join(aux, "holidays.json"), encoding="utf-8-sig"))
    days = {pd.to_datetime(e["dtstart"][0]).date() for e in h["vcalendar"][0]["vevent"]}
    return days | {pd.Timestamp(d).date() for d in HOLIDAYS_2023_2024}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", default="data/processed/ae_snapshots.csv")
    ap.add_argument("--aux", default="data/raw/aux")
    ap.add_argument("--out", default="data/processed/hourly_features.csv")
    a = ap.parse_args()
    ae = pd.read_csv(a.ae, parse_dates=["ts"])
    ae["hour_ts"] = ae["ts"].dt.floor("h")
    h = (ae.groupby(["hospital", "fmt", "hour_ts"], as_index=False)
           .agg(t45_min=("t45_min", "mean"), t45_over2h=("t45_over2h", "max"),
                t3p50_min=("t3p50_min", "mean"), n_snap=("ts", "count")))
    h["dow"] = h.hour_ts.dt.dayofweek; h["hr"] = h.hour_ts.dt.hour; h["month"] = h.hour_ts.dt.month
    h["is_weekend"] = (h.dow >= 5).astype(int)
    hol = load_holidays(a.aux)
    h["is_holiday"] = h.hour_ts.dt.date.isin(hol).astype(int)
    aq = load_aqhi(a.aux)
    h["station"] = h.hospital.map(HOSP_TO_STATION)
    h = h.merge(aq.rename(columns={"ts": "hour_ts"}), on=["hour_ts", "station"], how="left")
    w = load_hko(a.aux)
    if w is not None:
        h["date"] = h.hour_ts.dt.normalize()
        h = h.merge(w, on="date", how="left").drop(columns="date")
    h = h.sort_values(["hospital", "hour_ts"])
    g = h.groupby("hospital")["t45_min"]
    h["t45_lag1"] = g.shift(1); h["t45_lag24"] = g.shift(24)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    h.to_csv(a.out, index=False)
    print(f"hourly rows: {len(h):,}  hospitals: {h.hospital.nunique()}  range: {h.hour_ts.min()} .. {h.hour_ts.max()}")
    print("missing share:", h[["t45_min", "aqhi", "rf_mm", "temp_mean", "is_holiday"]].isna().mean().round(3).to_dict())
    print("holiday hours:", int(h.is_holiday.sum()), "| unmapped stations:", int(h.station.isna().sum()))


if __name__ == "__main__":
    main()
