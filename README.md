# DS5020 Final Project — Bayesian A/B Testing (Project 7)

**Iterations 2 & 3 | Fall 2026**

Bayesian A/B testing with the Beta–Binomial model, Monte Carlo estimation, numerical integration,
and the pooled two-proportion z-test, applied to a known-truth simulation and to the Cookie Cats
experiment. The core methods are written with NumPy only; SciPy is used **only to check** them.

## Folder structure

```text
├── data/                     # cookie_cats.csv goes here (NOT tracked by Git, see "Dataset")
├── docs/
│   ├── Iteration2and3_Report_Bayesian_AB_Testing.pdf   # combined Overleaf report
│   ├── Iteration2_Report_Bayesian_AB_Testing.pdf       # Iteration 2 report (as submitted)
│   ├── Project_Plan_Iterations_2_and_3.xlsx            # task table, timeline, contributions
│   ├── main.tex                                        # LaTeX source of the combined report
│   ├── verification_table.(csv|tex)                    # Iteration 3 verification table
│   └── *.csv, *.txt                                    # outputs written by the scripts
├── figures/                  # all plots (Iteration 2 plots + error_convergence.png)
├── src/
│   ├── methods.py            # Iteration 3: one my_* function per algorithm step (NumPy only)
│   ├── simulate.py           # Iteration 3: simulate_ab(n, seed) -> data + true parameters
│   ├── verify.py             # report() helper and verification-table logger
│   ├── library_checks.py     # SciPy reference implementations (check only)
│   ├── small_example.py      # Iteration 3: tiny worked example, step by step
│   ├── check_simulation.py   # Iteration 3: simulation test, error vs n, stress tests
│   ├── run_real.py           # Iteration 3: Cookie Cats analysis + library comparison
│   ├── make_verification_table.py   # merges the three verification CSVs
│   └── common.py, simulation.py, real_data.py, hand_example.py, run_all.py
│                             # Iteration 2 code (kept as submitted; run with `python -m src.<name>`)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset

Download the Kaggle **Cookie Cats A/B Testing** dataset and save it as `data/cookie_cats.csv`
(columns `userid, version, sum_gamerounds, retention_1, retention_7`). The raw file is ignored by Git
(`.gitignore`), so every member downloads it once.

## Run (from the repository root)

**Iteration 3**

```bash
python src/small_example.py          # tiny example, every step printed, exact fractions
python src/check_simulation.py       # simulation test, error-vs-n plot, stress tests
python src/run_real.py               # Cookie Cats analysis, cleaning log, SciPy comparison
python src/make_verification_table.py   # writes docs/verification_table.(csv|tex)
```

**Iteration 2** (unchanged)

```bash
python -m src.run_all --csv data/cookie_cats.csv
python -m src.hand_example
```

## Reproducibility

* Every random number comes from `numpy.random.default_rng(seed)`; seed 0 is the default.
* The simulation uses the Iteration 2 model: `p_A = 0.10`, `p_B = 0.12`, `n = 1000` per group, seed 0,
  `x_A ~ Binomial(n, p_A)` then `x_B ~ Binomial(n, p_B)`. This gives `x_A = 92, x_B = 121` in both iterations
  (`check_simulation.py` asserts it).
* Every number in the verification table (`docs/verification_table.csv`) is produced by the scripts above.
* Each script prints a PASS/FAIL line per check; `make_verification_table.py` reports the number of failures.

## Team

Kamya Kanara, Rishita Chawla, Vaibhavi. Iteration 2 leader: Kamya Kanara. Iteration 3 leader: Rishita Chawla.
Ownership, checkers and contributions are recorded in `docs/Project_Plan_Iterations_2_and_3.xlsx`.
