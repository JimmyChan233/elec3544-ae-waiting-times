"""Save the current A&E waiting-time snapshot, as a backup to the data.gov.hk archive for the November live test.

Usage: python src/pull_live.py [--out ~/elec3544_data/raw/ae]   (run every 15 min from cron / launchd)

Why: the historical archive has whole days with zero versions (Oct 2026: 5, 7, 8, 12, 16, 25, 26 are empty),
so the live-test window cannot rely on it alone. Files go to <out>/new/<YYYYMMDD-HHMM>.json, the same layout
parse_ae.py reads; the timestamp is the feed's own updateTime rounded down to 15 min, so re-runs never duplicate.
"""
import argparse, datetime as dt, json, os, subprocess

URL = "https://www.ha.org.hk/opendata/aed/aedwtdata2-en.json"
UA = "Mozilla/5.0 (HKU ELEC3544 Group 16 student project)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/elec3544_data/raw/ae"))
    a = ap.parse_args()
    r = subprocess.run(["curl", "-sL", "--max-time", "60", "-A", UA, URL], capture_output=True, check=True)
    body = json.loads(r.stdout)
    t = dt.datetime.strptime(body["updateTime"], "%d/%m/%Y %I:%M%p")
    t = t.replace(minute=t.minute // 15 * 15)
    path = os.path.join(a.out, "new", t.strftime("%Y%m%d-%H%M") + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(r.stdout)
    print(path)


if __name__ == "__main__":
    main()
