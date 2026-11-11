"""Download Hospital Authority A&E waiting-time snapshots from the data.gov.hk historical archive.

Usage:
  python src/download_ae_archive.py --start 20240101 --end 20240107 [--every 1] [--out data/raw/ae]

The archive keeps every 15-minute snapshot. The feed URL changed on 2025-10-13:
  old (single categorical field "topWait")      : aedwtdata-en.json   (archive from 2023-01)
  new (per-triage p50/p95 waits as text)         : aedwtdata2-en.json  (from 2025-10-13 09:30)
Files are saved as <out>/<old|new>/<YYYYMMDD-HHMM>.json and the script is resumable (existing files are skipped).
list-file-versions truncates at ~35k results, so we query one calendar month at a time.
"""
import argparse, calendar, datetime as dt, json, os, subprocess, sys, time

OLD_URL = "https://www.ha.org.hk/opendata/aed/aedwtdata-en.json"
NEW_URL = "https://www.ha.org.hk/opendata/aed/aedwtdata2-en.json"
SWITCH = dt.datetime(2025, 10, 13, 9, 30)
LIST = "https://api.data.gov.hk/v1/historical-archive/list-file-versions?url={url}&start={s}&end={e}"
GET = "https://api.data.gov.hk/v1/historical-archive/get-file?url={url}&time={t}"
UA = "Mozilla/5.0 (HKU ELEC3544 Group 16 student project)"


def curl(url, timeout=60, tries=4):
    for i in range(tries):
        r = subprocess.run(["curl", "-sL", "--max-time", str(timeout), "-A", UA, url], capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"curl failed: {url}")


def month_windows(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        last = dt.date(cur.year, cur.month, calendar.monthrange(cur.year, cur.month)[1])
        yield cur, min(last, end)
        cur = last + dt.timedelta(days=1)


def list_versions(url, s: dt.date, e: dt.date):
    # the API rejects an end date of today or later, so cap at yesterday; add a day otherwise, then filter.
    e_api = min(e + dt.timedelta(days=1), dt.date.today() - dt.timedelta(days=1))
    raw = curl(LIST.format(url=url, s=s.strftime("%Y%m%d"), e=e_api.strftime("%Y%m%d")))
    ts = json.loads(raw).get("timestamps", [])
    return [t for t in ts if s.strftime("%Y%m%d") <= t[:8] <= e.strftime("%Y%m%d")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--every", type=int, default=1, help="keep every Nth snapshot (4 = hourly)")
    ap.add_argument("--out", default="data/raw/ae")
    ap.add_argument("--sleep", type=float, default=0.15)
    a = ap.parse_args()
    start = dt.datetime.strptime(a.start, "%Y%m%d").date()
    end = dt.datetime.strptime(a.end, "%Y%m%d").date()
    total = saved = skipped = 0
    for s, e in month_windows(start, end):
        for fmt, url in (("old", OLD_URL), ("new", NEW_URL)):
            ts = list_versions(url, s, e)
            ts = [t for t in ts if (dt.datetime.strptime(t, "%Y%m%d-%H%M") < SWITCH) == (fmt == "old")]
            ts = ts[:: a.every]
            os.makedirs(os.path.join(a.out, fmt), exist_ok=True)
            print(f"[{s}..{e}] {fmt}: {len(ts)} snapshots", flush=True)
            for i, t in enumerate(ts):
                total += 1
                path = os.path.join(a.out, fmt, f"{t}.json")
                if os.path.exists(path) and os.path.getsize(path) > 50:
                    skipped += 1
                    continue
                try:
                    body = curl(GET.format(url=url, t=t))
                    json.loads(body)  # validate
                except Exception as ex:
                    print(f"  ! {t}: {ex}", file=sys.stderr, flush=True)
                    continue
                with open(path, "wb") as f:
                    f.write(body)
                saved += 1
                if saved % 100 == 0:
                    print(f"  saved {saved} (skipped {skipped}) last={t}", flush=True)
                time.sleep(a.sleep)
    print(f"done: total {total}, saved {saved}, skipped {skipped}")


if __name__ == "__main__":
    main()
