"""Fetch auxiliary feature sources: HK public holidays, EPD hourly AQHI, HKO daily weather.

Usage: python src/fetch_aux.py [--out data/raw/aux] [--aqhi-from 2023-01] [--aqhi-to 2026-10]
Writes:
  holidays.json                     1823.gov.hk iCal-as-JSON (current + next year)
  aqhi/YYYYMM_Eng.csv               EPD hourly AQHI per station, one file per month
  hko/daily_HKO_<ELE>_ALL.csv       HKO daily series (rainfall RF, mean RH, temperature)
"""
import argparse, os, subprocess, time
import pandas as pd

UA = "Mozilla/5.0 (HKU ELEC3544 Group 16 student project)"
HOLIDAYS = "https://www.1823.gov.hk/common/ical/en.json"
AQHI = "https://www.aqhi.gov.hk/epd/ddata/html/history/{y}/{ym}_Eng.csv"
HKO = "https://data.weather.gov.hk/weatherAPI/cis/csvfile/HKO/ALL/daily_HKO_{ele}_ALL.csv"
HKO_ELEMENTS = ["RF", "RH", "TEMP", "MAXT", "MINT"]


def fetch(url, path, min_bytes=200):
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        return "skip"
    r = subprocess.run(["curl", "-sL", "--max-time", "90", "-A", UA, url], capture_output=True)
    body = r.stdout
    if r.returncode != 0 or len(body) < min_bytes or body.lstrip()[:5].lower() == b"<html":
        return "FAIL"
    with open(path, "wb") as f:
        f.write(body)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/aux")
    ap.add_argument("--aqhi-from", default="2023-01")
    ap.add_argument("--aqhi-to", default="2026-10")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "aqhi"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "hko"), exist_ok=True)
    print("holidays:", fetch(HOLIDAYS, os.path.join(a.out, "holidays.json")))
    for ele in HKO_ELEMENTS:
        print(f"hko {ele}:", fetch(HKO.format(ele=ele), os.path.join(a.out, "hko", f"daily_HKO_{ele}_ALL.csv")))
    for p in pd.period_range(a.aqhi_from, a.aqhi_to, freq="M"):
        ym = p.strftime("%Y%m")
        st = fetch(AQHI.format(y=p.year, ym=ym), os.path.join(a.out, "aqhi", f"{ym}_Eng.csv"))
        print(f"aqhi {ym}: {st}")
        if st == "ok":
            time.sleep(0.3)


if __name__ == "__main__":
    main()
