#!/bin/zsh
# Full-history hourly download (every 4th 15-min snapshot), 8 parallel workers, raw files kept OUTSIDE iCloud.
# Re-running is safe: existing files are skipped. Logs: ~/elec3544_data/logs/worker_N.log
set -u
D="$HOME/elec3544_data"; mkdir -p "$D/raw/ae" "$D/logs"
cd "$(dirname "$0")"
# month ranges (start end) grouped into 8 workers
typeset -a W
W=( "20230101 20230630" "20230701 20231231" "20240101 20240630" "20240701 20241231" \
    "20250101 20250630" "20250701 20251231" "20260101 20260630" "20260701 20261031" )
i=0
for r in "${W[@]}"; do
  i=$((i+1)); s=${r%% *}; e=${r##* }
  nohup python3 src/download_ae_archive.py --start $s --end $e --every 4 --sleep 0.05 --out "$D/raw/ae" \
      > "$D/logs/worker_$i.log" 2>&1 &
  echo "worker $i: $s..$e pid $!"
done
echo "progress: ls ~/elec3544_data/raw/ae/old | wc -l ; ls ~/elec3544_data/raw/ae/new | wc -l   (target ~33,000 total)"
