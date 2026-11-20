
# DS5020 Final Project — Bayesian A/B Testing (Project 7)

**Iterations 2 & 3 | Fall 2026**

Bayesian A/B testing using the Beta–Binomial model, Monte Carlo estimation, numerical integration, and the two-proportion z-test. The project includes a worked example, simulations, numerical verification, and analysis of the Cookie Cats dataset.


## Folder Structure

```text
├── data/
│   └── cookie_cats.csv
├── docs/
│   ├── Iteration2_Report_Bayesian_AB_Testing.pdf
│   │   # Iteration 2 report
│   ├── Iteration2and3_Report_Bayesian_AB_Testing.pdf
│   │   # Combined Iteration 2 & 3 report
│   ├── Project_Plan_Iterations_2_and_3.xlsx
│   │   # Combined Iteration 2 & 3 project plan
│   ├── main.tex
│   │   # LaTeX source
│   └── *.csv
│       # Simulation and real-data results
├── figures/        # Iteration 2 plots and error_convergence plot from iteration 3
├── src/            # Python scripts and verification tests
├── requirements.txt
└── README.md
```


## Setup

Run from the project root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

**Iteration 2:**

```bash
python -m src.run_all --csv data/cookie_cats.csv
```

**Iteration 3:**

```bash
python src/small_example.py       # Worked example and math verification
python src/check_simulation.py    # Simulation, convergence, stress tests
python src/run_real.py            # Cookie Cats analysis and verification
```

## Dataset

Download the Kaggle **Cookie Cats A/B Testing** dataset and save it as `data/cookie_cats.csv`.

Required columns: `userid`, `version`, `sum_gamerounds`, `retention_1`, `retention_7`.

The raw dataset is not tracked by Git.

## Reproducibility

Simulations use fixed random seeds. Iteration 3 uses `p_A = 0.10`, `p_B = 0.12`, `n = 1000` per group, and `seed = 0`.

The implementation is verified against SciPy. Results, mathematical explanations, and team contributions are documented in the Overleaf report and Excel project plan in `docs/`.
