"""Typeset the report in the official ELEC3544 LaTeX template (course style file, pdfLaTeX, BibTeX).

Called by:  python3 report/build_final.py --official
Source text: report/report_official.md (numbers filled in from results/ by build_final.py)
Template:    report/template/elec3544_report.sty, the unchanged course style file from the Moodle template zip
Output:      ../Assignments/Report/ELEC3544-Report-Group16.pdf
"""
import re, shutil, subprocess
from pathlib import Path

TITLE = "Predicting A\\&E Waiting Times in Hong Kong Public Hospitals"
AUTHORS = ["CHEN Hongxi", "NI Haoran", "ZHU Yongkai"]
PREAMBLE = r"""\documentclass[10pt,letterpaper]{article}
\usepackage{elec3544_report}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,nicefrac,xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype,needspace}
% needed by the pandoc-generated body
\usepackage{longtable,array,calc,float,enumitem,fancyvrb}
\floatplacement{figure}{!htb}
\setlist{itemsep=1pt,topsep=2pt,parsep=0pt,leftmargin=1.4em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\newcounter{none} % pandoc uses it for tables without a caption
\setlength{\LTpre}{4pt}\setlength{\LTpost}{4pt}
\AtBeginEnvironment{longtable}{\footnotesize}
\DeclareUnicodeCharacter{2212}{\ensuremath{-}}
\DeclareUnicodeCharacter{00B2}{\ensuremath{^2}}
\DeclareUnicodeCharacter{00D7}{\ensuremath{\times}}
\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}
\DeclareUnicodeCharacter{2192}{\ensuremath{\to}}
\DeclareUnicodeCharacter{2248}{\ensuremath{\approx}}
\DeclareUnicodeCharacter{2260}{\ensuremath{\neq}}
\RecustomVerbatimEnvironment{verbatim}{Verbatim}{fontsize=\scriptsize}
"""


def build(N, ROOT, OUT):
    local = ROOT / "report/template/elec3544_report.sty"   # the course style file; get it from the Moodle template zip
    tpl = local if local.exists() else next((ROOT.parent / "Assignments" / "Official-Template-20261111").rglob("elec3544_report.sty"))
    B = Path.home() / "elec3544_data" / "build_official"          # no spaces in the path: pdfLaTeX is fussy
    shutil.rmtree(B, ignore_errors=True); B.mkdir(parents=True)
    shutil.copy(tpl, B); shutil.copy(ROOT / "report/references.bib", B)
    N = dict(N)
    N["table_post"] = N["table_post"].replace(" [10]", "")   # numeric marker from the old layout; cited in the text instead
    for k in [k for k in N if k.startswith("fig_")]:
        shutil.copy(N[k], B / f"{k}.png"); N[k] = f"{k}.png"
    md = B / "body.md"
    text = (ROOT / "report/report_official.md").read_text().format_map(N)
    md.write_text(text)
    m = re.search(r"\\begin\{abstract\}\n(.*?)\n\\end\{abstract\}\n", text, re.S)   # abstract goes through pandoc too, so & and % are escaped
    tex = lambda t: subprocess.run(["pandoc", "-f", "markdown+raw_tex", "-t", "latex", "--wrap=none"], input=t, capture_output=True, text=True, check=True).stdout
    body = "\\begin{abstract}\n" + tex(m.group(1)) + "\\end{abstract}\n\n" + tex(text[m.end():])
    (B / "body.tex").write_text(body)
    author = "Group 16\\\\\n" + "\\\\\n".join(AUTHORS)
    (B / "main.tex").write_text(PREAMBLE + f"\\title{{{TITLE}}}\n\\author{{{author}}}\n\\begin{{document}}\n\\maketitle\n" + body + "\n\\end{document}\n")
    run = lambda *c: subprocess.run(c, cwd=B, capture_output=True, text=True)
    run("pdflatex", "-interaction=nonstopmode", "main"); run("bibtex", "main")
    run("pdflatex", "-interaction=nonstopmode", "main"); r = run("pdflatex", "-interaction=nonstopmode", "main")
    log = (B / "main.log").read_text(errors="ignore")
    bad = [l for l in log.split("\n") if l.startswith("!") or "Undefined control" in l or "Citation" in l and "undefined" in l or "Missing character" in l]
    over = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)
    print("LaTeX problems:", bad[:8] or "none", "| overfull boxes > 5pt:", sum(float(x) > 5 for x in over))
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(B / "main.pdf", OUT / "ELEC3544-Report-Group16.pdf")
    shutil.copy(md, OUT / "ELEC3544-Report-Group16.md")
    print(f"wrote {OUT / 'ELEC3544-Report-Group16.pdf'} (official template)")
