#!/bin/zsh
# Waits for all download workers to exit, then parses the full history and runs features + EDA/baselines.
# Output: results/eda_baseline_full_<date>.txt
cd "$(dirname "$0")"
while pgrep -f download_ae_archive >/dev/null; do sleep 60; done
D="$HOME/elec3544_data/raw/ae"
echo "downloads finished $(date): old $(ls $D/old | wc -l) new $(ls $D/new | wc -l)"
python3 src/parse_ae.py --raw "$D" --out data/processed/ae_snapshots_full.csv
python3 src/build_features.py --ae data/processed/ae_snapshots_full.csv --out data/processed/hourly_features_full.csv
python3 src/eda_baseline.py --in data/processed/ae_snapshots_full.csv
echo "ALL_DONE $(date)"
