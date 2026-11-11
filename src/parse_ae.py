"""Parse raw A&E snapshots (both formats) into one tidy table.

Usage: python src/parse_ae.py [--raw data/raw/ae] [--out data/processed/ae_snapshots.csv]

Output columns:
  ts (snapshot time, HKT), hospital, fmt (old|new),
  t45_min   : waiting time for triage cat 4/5 in minutes (old: lower bound of category; new: p50)
  t45_over2h: 1 if cat 4/5 wait > 2 hours (available in BOTH formats)
  t3p50_min, t3p95_min, t45p95_min, t1_min, t2_min : new format only (NaN for old)
"""
import argparse, glob, json, os, re
import pandas as pd

_NUM = r"(\d+(?:\.\d+)?)"


def parse_wait(s):
    """Return (minutes, kind) where kind in {'exact','lt','over','around'}; (None,None) if unparseable.

    'less than 15 minutes' -> (7.5,'lt')  midpoint of [0,15)
    '0 minute' / '23 minutes' -> (0,'exact') / (23,'exact')
    '1 hour' / '3 hour' / '2.5 hours' -> (60,'exact') / (180,'exact') / (150,'exact')
    'Around 1 hour' -> (60,'around')
    'Over 3 hours' -> (180,'over')  lower bound of the category
    """
    if s is None:
        return None, None
    t = str(s).strip().lower()
    if not t:
        return None, None
    m = re.fullmatch(r"less than " + _NUM + r" minutes?", t)
    if m:
        return float(m.group(1)) / 2, "lt"
    m = re.fullmatch(r"(over|more than) " + _NUM + r" hours?", t)
    if m:
        return float(m.group(2)) * 60, "over"
    m = re.fullmatch(r"around " + _NUM + r" hours?", t)
    if m:
        return float(m.group(1)) * 60, "around"
    m = re.fullmatch(_NUM + r" minutes?", t)
    if m:
        return float(m.group(1)), "exact"
    m = re.fullmatch(_NUM + r" hours?", t)
    if m:
        return float(m.group(1)) * 60, "exact"
    return None, None


def over_2h(minutes, kind):
    """Cat 4/5 wait strictly longer than 2 hours. 'Over 2 hours' (lower bound 120) counts as >2h."""
    if minutes is None:
        return None
    if kind == "over":
        return int(minutes >= 120)
    return int(minutes > 120)


def parse_file(path, fmt):
    ts = pd.to_datetime(os.path.basename(path)[:13], format="%Y%m%d-%H%M")
    with open(path, "rb") as f:
        d = json.load(f)
    rows = []
    for h in d.get("waitTime", []):
        r = {"ts": ts, "hospital": h.get("hospName"), "fmt": fmt}
        if fmt == "old":
            mins, kind = parse_wait(h.get("topWait"))
            r["t45_min"], r["t45_kind"] = mins, kind
            r["t45_over2h"] = over_2h(mins, kind)
        else:
            mins, kind = parse_wait(h.get("t45p50"))
            r["t45_min"], r["t45_kind"] = mins, kind
            r["t45_over2h"] = over_2h(mins, kind)
            for col, key in (("t3p50_min", "t3p50"), ("t3p95_min", "t3p95"), ("t45p95_min", "t45p95"),
                             ("t1_min", "t1wt"), ("t2_min", "t2wt")):
                r[col] = parse_wait(h.get(key))[0]
        rows.append(r)
    return rows


def build(raw_dir):
    rows = []
    for fmt in ("old", "new"):
        for p in sorted(glob.glob(os.path.join(raw_dir, fmt, "*.json"))):
            rows.extend(parse_file(p, fmt))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["ts", "hospital"]).drop_duplicates(["ts", "hospital"])
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/ae")
    ap.add_argument("--out", default="data/processed/ae_snapshots.csv")
    a = ap.parse_args()
    df = build(a.raw)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"rows={len(df)} snapshots={df['ts'].nunique() if len(df) else 0} hospitals={df['hospital'].nunique() if len(df) else 0}")
    if len(df):
        print("unparsed t45:", int(df["t45_min"].isna().sum()))
        print(df.groupby("fmt").size())
