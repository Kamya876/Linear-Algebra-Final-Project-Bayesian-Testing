"""Merge the per-script verification CSVs into one table (CSV + LaTeX).

Run AFTER small_example.py, check_simulation.py and run_real.py.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify import DOCS  # noqa: E402

PARTS = ["verification_small_example.csv", "verification_simulation.csv", "verification_real.csv"]


def tex_escape(s):
    return (s.replace("\\", "").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
             .replace("^", r"\textasciicircum{}").replace("|", r"$|$").replace("<=", r"$\le$").replace(">=", r"$\ge$"))


def main():
    rows = []
    for name in PARTS:
        path = DOCS / name
        if not path.exists():
            raise SystemExit(f"Missing {path}; run the scripts first.")
        with open(path) as f:
            rows += list(csv.reader(f))[1:]
    with open(DOCS / "verification_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Data", "Quantity", "Compared with", "Difference", "Pass?"])
        w.writerows(rows)
    with open(DOCS / "verification_table.tex", "w") as f:
        head = "Data & Quantity & Compared with & Difference & Pass?\\\\\n\\midrule\n"
        f.write("\\begin{longtable}{>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{3.6cm}>{\\raggedright\\arraybackslash}p{3.4cm}>{\\raggedright\\arraybackslash}p{3.9cm}l}\n")
        f.write("\\caption{Verification table. Every number is produced by \\texttt{src/small\\_example.py}, "
                "\\texttt{src/check\\_simulation.py} or \\texttt{src/run\\_real.py}.}\\label{tab:verification}\\\\\n")
        f.write("\\toprule\n" + head + "\\endfirsthead\n\\toprule\n" + head + "\\endhead\n")
        for r in rows:
            mark = r"\checkmark" if r[4] == "PASS" else r"\textbf{FAIL}"
            f.write(" & ".join(tex_escape(c) for c in r[:4]) + f" & {mark}\\\\\n")
        f.write("\\bottomrule\n\\end{longtable}\n")
    n_fail = sum(r[4] != "PASS" for r in rows)
    print(f"Verification table: {len(rows)} rows, {n_fail} failures -> docs/verification_table.(csv|tex)")


if __name__ == "__main__":
    main()
