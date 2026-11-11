#!/bin/zsh
# Collect the final deliverables into ELEC3544/Final-Submission (copies; the working files stay where they are).
# Run after any rebuild:  ./make_submission.sh
set -e
cd "$(dirname "$0")"
A=../Assignments
OUT=../Final-Submission
rm -rf "$OUT"
mkdir -p "$OUT/1-Upload-to-Moodle" "$OUT/2-Presentation" "$OUT/3-Code-and-Data" "$OUT/4-Course-Requirements"

cp "$A/Report/ELEC3544-Report-Group16.pdf"  "$OUT/1-Upload-to-Moodle/"
cp "$A/Slides/ELEC3544-Slides-Group16.pdf"  "$OUT/1-Upload-to-Moodle/"

cp "$A/Slides/ELEC3544-Slides-Group16.html" "$OUT/2-Presentation/Slides-to-present.html"
cp "$A/Slides/ELEC3544-Slides-Group16.pdf"  "$OUT/2-Presentation/Slides-backup.pdf"
cp "$A/Slides/presentation-script.md"       "$OUT/2-Presentation/"
[ -f "$A/Slides/presentation-script.pdf" ] && cp "$A/Slides/presentation-script.pdf" "$OUT/2-Presentation/"

cp "$A/Share/Group16-Project-2026-11-11.zip" "$OUT/3-Code-and-Data/"
cp EXPERIMENT_LOG.md README.md               "$OUT/3-Code-and-Data/"

cp "$A/ELEC3544-GroupProject-ReportSlides-Guide.pdf" "$A/ELEC3544-GroupProject-AssessmentStandard.pdf" \
   "$A/ELEC3544-GroupProject-Presentation-Timetable.pdf" "$A/ELEC3544-ProjectProposal-Group16.pdf" "$OUT/4-Course-Requirements/"

cat > "$OUT/README.md" <<EOF
# ELEC3544 Group 16: final submission

Built $(date "+%d %b %Y %H:%M"). Everything here is a copy; edit the sources, then run \`Project/make_submission.sh\` again.

## 1-Upload-to-Moodle
Each member uploads **both** PDFs to "Group Project Report and Slides" by **Tue 17 Nov 2026, 23:59**.
- \`ELEC3544-Report-Group16.pdf\`: the report, $(pdfinfo "$A/Report/ELEC3544-Report-Group16.pdf" | awk '/Pages/{print $2}') pages (limit 8).
- \`ELEC3544-Slides-Group16.pdf\`: the slides, 14 main + 3 backup.

## 2-Presentation
Wed 18 Nov 2026, 16:45-16:55, CPD-1.21. Ten-minute hard cutoff; every member speaks.
- \`Slides-to-present.html\`: open in Chrome, arrow keys or click to move, \`f\` for full screen. Needs internet once for the fonts.
- \`Slides-backup.pdf\`: use this if the HTML misbehaves.
- \`presentation-script\`: what each person says, with timings and likely questions.

## 3-Code-and-Data
- \`Group16-Project-2026-11-11.zip\`: code, processed data and every result file. Its SHA-256 prefix is printed in the report.
- \`EXPERIMENT_LOG.md\`, \`README.md\`: copies from inside the zip, for quick reading.

## 4-Course-Requirements
The brief, the rubric, the timetable and our proposal.

## Where the sources are
- Report text: \`Project/report/report_final.md\`, built by \`Project/report/build_final.py\`.
- Slides: \`Assignments/Slides/build_deck.py\`, rendered by \`Assignments/Slides/render.sh\`.
- Script: \`Assignments/Slides/presentation-script.md\`.
EOF
echo "Final-Submission rebuilt:"; find "$OUT" -type f | sort
