#!/bin/zsh
# Export a clean copy of the project to ~/elec3544_data/repo (outside iCloud) for GitHub.
# Leaves out: large processed tables (shipped as a release asset), caches, old drafts, student numbers.
set -e
cd "$(dirname "$0")"
DEST="$HOME/elec3544_data/repo"
mkdir -p "$DEST"
rsync -a --delete --exclude '.git' --exclude '.DS_Store' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude 'notebooks' --exclude 'data/processed' --exclude 'data/raw/ae' --exclude 'data/raw/aux/_old*' \
  --exclude 'results/review_packet.md' --exclude 'report/report_template.md' --exclude 'report/template' \
  ./ "$DEST/"
# names stay, student numbers do not go into a public repository
sed -i '' -E 's/ \(30[0-9]{8}\)//g' "$DEST/report/report_final.md" "$DEST/report/report_official.md" "$DEST/report/build_official.py"
printf 'The course style file elec3544_report.sty is not redistributed here. Take it from the ELEC3544 report template on Moodle and put it in this folder before running report/build_final.py --official.\n' > "$DEST/report/template-README.txt"
cat > "$DEST/.gitignore" <<'G'
__pycache__/
.pytest_cache/
.DS_Store
data/processed/*.csv
data/raw/ae/
*.zip
G
mkdir -p "$DEST/data/processed"
cat > "$DEST/data/processed/README.md" <<'G'
The processed tables (`hourly_features_full.csv`, `hourly_features_latest.csv`, about 70 MB each) are too
large for the repository. Download `Group16-Project-data.zip` from the repository's Releases page and unzip it
here, or rebuild them from the public archive with the commands in the top-level README.
G
if grep -rIl -E "30[0-9]{8}" "$DEST" --exclude-dir=.git --exclude='*.csv' --exclude='*.json' | grep -q .; then
  echo "WARNING: a student number is still present:"; grep -rIl -E "30[0-9]{8}" "$DEST" --exclude-dir=.git --exclude='*.csv' --exclude='*.json'
fi
echo "exported to $DEST: $(du -sh "$DEST" | cut -f1), $(find "$DEST" -type f -not -path '*/.git/*' | wc -l | tr -d ' ') files"
