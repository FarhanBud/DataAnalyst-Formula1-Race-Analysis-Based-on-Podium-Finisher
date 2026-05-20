# F1 Podium Prediction: A Machine Learning Approach to Race Outcome Forecasting

## 📌 Executive Summary

**Predicting the Podium: Data-Driven Race Intelligence Across 75 Years of Formula 1**

This project develops a predictive machine learning model to classify which Formula 1 drivers will finish on the podium (Top 3) in a given Grand Prix. By analyzing historical race data spanning 1950 to 2026 — covering qualifying performance, grid positions, driver form, and constructor identity — the model transforms raw race statistics into actionable pre-race intelligence for analysts, strategists, and enthusiasts.

**Key Achievement:** XGBoost classifier achieving **ROC-AUC of 0.9738** and **Podium F1-Score of 0.8678**, correctly identifying 614 out of 699 true podium finishes on a held-out test set of 3,137 entries.

---

## 🎯 Business Problem

### The Challenge

Formula 1 is one of the most data-rich sports in the world, yet podium prediction remains highly complex due to:

- **Field Size:** 20 drivers per race, only 3 podium spots (15% theoretical maximum)
- **Historical Class Imbalance:** Only 22.29% of all race entries result in a podium finish
- **Era Complexity:** Racing regulations, qualifying formats, and constructor dominance have changed dramatically across 75 years
- **Multivariate Dependencies:** Grid position, qualifying pace, recent driver form, team identity, and circuit characteristics all interact simultaneously

### Current State Without Predictive Modeling
- Race analysts rely on subjective expert opinion without probabilistic grounding
- No systematic framework to quantify podium probability per driver per race
- Fantasy F1 players, broadcasters, and team strategists lack an objective pre-race baseline
- Historical performance patterns of constructors at specific circuits are not systematically leveraged

### Root Cause Analysis
1. **Data Fragmentation:** Race results, qualifying times, driver standings, and circuit data exist as separate tables — requiring significant integration work before any analysis is possible
2. **Temporal Complexity:** Driver momentum, team form, and circuit-specific performance evolve across seasons and are difficult to capture without feature engineering
3. **Class Imbalance:** Naive models default to predicting "Non-Podium" for all entries and achieve 77.7% accuracy while being completely useless for race prediction
4. **Qualifying Data Sparsity:** Pre-2006 races predate the Q1/Q2/Q3 format, creating gaps in one of the most predictive feature categories

---

## 🎯 Project Objectives

### Primary Goals
1. **Predictive Modeling:** Build a binary classification model (`is_podium`) to predict race podium finishes with high discriminative power
2. **Feature Insight:** Identify which factors — grid position, qualifying delta, driver momentum, constructor identity — most strongly predict podium outcomes
3. **Temporal Analysis:** Quantify how podium predictability has evolved across three distinct eras of Formula 1 history
4. **Actionable Output:** Produce a race-ready prediction dataset with confidence scoring for integration into a Tableau monitoring dashboard

### Success Metrics
- **ROC-AUC ≥ 0.95:** Strong class discrimination beyond simple accuracy
- **Podium F1-Score ≥ 0.80:** Balanced precision and recall on the minority class
- **Recall ≥ 85%:** Detect the majority of true podium finishes to minimize missed predictions
- **Uncertainty Flagging:** Label predictions with confidence levels for operational use

---

## 📊 Dataset Overview

### Source
- **Ergast Developer API:** [ergast.com/mrd](http://ergast.com/mrd/) — Official F1 historical data archive
- **Coverage:** Every official Formula 1 World Championship race from 1950 through the 2026 season

### Dataset Characteristics

| Attribute | Value |
|---|---|
| **Total Race Entries** | 15,681 |
| **Seasons Covered** | 1950 – 2026 (75 years) |
| **Unique Drivers** | 650 |
| **Unique Constructors** | 162 |
| **Unique Circuits** | 77 |
| **Target Variable** | `is_podium` (Binary: 0 = Non-Podium, 1 = Podium) |
| **Class Distribution** | 77.71% Non-Podium / 22.29% Podium |
| **Source Tables Merged** | 8 (results, races, drivers, constructors, qualifying, circuits, driver standings, constructor standings) |

### Era Breakdown

| Era | Entries | Podiums | Podium Rate |
|---|---|---|---|
| Classic (1950–1979) | 3,772 | 1,028 | 27.3% |
| Pre-Q3 Format (1980–2005) | 5,068 | 1,262 | 24.9% |
| Modern Q1/Q2/Q3 (2006–2026) | 6,841 | 1,206 | 17.6% |

> The declining podium rate across eras reflects the gradual expansion of grid sizes from ~20 to 26 drivers in the modern era, compressing individual podium probability.

### Key Features

**Race & Grid Variables:**
- `grid` — Starting grid position (strongest single numeric predictor; pole = 84.1% podium rate)
- `round` — Race number within the season
- `laps` — Total laps completed

**Qualifying Variables:**
- `qualifying_delta` — Time gap (ms) between driver's best qualifying lap and pole position time; derived using Q3 → Q2 → Q1 fallback strategy

**Driver Variables:**
- `driver_name` — Full driver name
- `driver_age` — Driver age at time of race (years)
- `driver_momentum` — Rolling points total from previous 3 races (captures in-season form)
- `nationality` — Driver nationality (650 drivers, 35+ nationalities)

**Constructor Variables:**
- `constructor` — Team name (162 unique constructors, one-hot encoded)
- `constructor_id` — Standardized team identifier

**Circuit Variables:**
- `circuit_id` — Standardized circuit identifier (77 circuits, one-hot encoded)
- `race_name` — Official Grand Prix name

**Target & Outcome Variables:**
- `is_podium` — Binary target (1 = P1/P2/P3, 0 = P4 and below)
- `position_num` — Actual finishing position (integer)
- `points` — Championship points awarded
- `status` — Race completion status (Finished, DNF, DSQ, etc.)

---

## 🔍 Key Findings from EDA

### Grid Position Analysis

**Grid Position vs Podium Rate (historical, all eras):**

| Starting Grid | Podium Rate | Interpretation |
|---|---|---|
| Grid 1 (Pole) | **84.1%** | Near-certainty of podium contention |
| Grid 2 | 76.3% | Front row still dominant |
| Grid 3 | 65.5% | Top 3 grid = strong expectation |
| Grid 4 | 49.8% | Below 50% for the first time |
| Grid 5 | 37.8% | Significant drop-off begins |
| Grid 10 | 9.5% | Single-digit probability |
| Grid 15 | 2.5% | Near-improbable without safety car/incident |
| Grid 20 | 1.4% | Essentially no historical precedent |

**Insight:** The relationship between grid position and podium probability is non-linear and steep — the top 3 grid positions account for a disproportionate share of all podiums. This validates the well-known F1 saying: *"Championships are won on Saturdays."*

---

### Constructor Dominance

**All-Time Podium Finishes by Constructor (Top 5):**

| Constructor | All-Time Podiums |
|---|---|
| Ferrari | 864 |
| McLaren | 543 |
| Mercedes | 315 |
| Williams | 316 |
| Red Bull | 297 |

**Insight:** Constructor identity carries persistent predictive power across eras. Ferrari's 864 all-time podiums represent over 15% of all podium slots in the sport's history — a level of structural dominance that no numeric feature alone can capture.

---

### Driver Nationality

**Top 5 Nationalities by Podium Count:**

| Nationality | Podiums |
|---|---|
| British | 809 |
| German | 415 |
| French | 318 |
| Brazilian | 292 |
| Finnish | 245 |

**Insight:** British drivers account for the most podiums in F1 history, reflecting both the sport's origins and the concentration of top teams (McLaren, Williams, Mercedes AMG) historically based in the UK.

---

### Qualifying Delta Feature

**Challenge identified and resolved:**
- The original `qualifying_delta` feature (gap to pole position in Q3 milliseconds) was meaningfully available only from 2006 onward when the Q1/Q2/Q3 format was introduced
- Pre-2006 entries and non-Q3 drivers received a maximum imputation value (31,189ms), causing 78.8% of all rows to share the exact same value — essentially noise
- **Fix applied:** Fallback strategy using Q3 → Q2 → Q1 → grid-position median, reducing the proportion of uniform-imputed values from 78.8% to under 6%

---

### Driver Momentum

**`driver_momentum` captures rolling 3-race points total:**
- Drivers with high momentum (15+ points in last 3 races) show meaningfully elevated podium rates in the next race
- This feature outperforms season-aggregate standings for short-term prediction, as it captures current form rather than accumulated historical totals
- Range in dataset: 0–31 championship points (3-race window)

---

## 🛠️ Methodology

### Data Integration Pipeline (8 Source Tables → 1 Master Dataset)

```
f1_results.csv          ─┐
f1_races.csv             │
f1_drivers.csv           ├──► MERGE (multi-key JOIN) ──► df_master
f1_constructors.csv      │         on raceId, driverId,
f1_qualifying.csv        │         constructorId
f1_circuits.csv          │
f1_driver_standings.csv  │
f1_constructor_standings.csv ─┘
```

### Data Cleaning Pipeline
1. **Schema Normalization:** Standardized ID columns, date formats, and categorical labels across 8 source files
2. **Qualifying Time Parsing:** Converted lap time strings (e.g., `"1:23.456"`) to milliseconds for numeric computation
3. **Qualifying Delta Fix:** Implemented Q3 → Q2 → Q1 fallback to replace 78.8% uniform imputation with meaningful pace data
4. **DNF/DSQ Filtering:** Excluded non-classified finishes from target variable assignment
5. **Column Pruning:** Removed `time`, `fastest_lap`, `fastest_lap_rank` (>46% missing values, no imputation viable)
6. **Missing Value Imputation:** `qualifying_delta` — grid-position median; `driver_age` — forward-fill by driver

### Feature Engineering
- **`qualifying_delta`:** Best qualifying time gap from pole (ms), with multi-session fallback
- **`driver_momentum`:** Rolling 3-race cumulative points per driver per season
- **`driver_age`:** Calculated from date of birth to race date (decimal years)
- **`is_podium`:** Binary target: 1 if `position_num` ∈ {1, 2, 3}, else 0
- **Constructor & Circuit Encoding:** One-hot encoded (161 constructor dummies + 76 circuit dummies = 241 total features after encoding)

### Exploratory Data Analysis
- Univariate distributions for all numerical and categorical features
- Bivariate analysis: feature impact on `is_podium`
- Era-segmented analysis: Classic vs Pre-Q3 vs Modern
- Grid position vs podium rate (non-linear relationship quantified)
- Constructor performance heatmap: top teams × top circuits
- SHAP summary plot: feature contribution direction and magnitude

### Class Imbalance Handling
**SMOTE (Synthetic Minority Oversampling Technique)** applied exclusively to training data:
- Original training split: 77.71% Non-Podium / 22.29% Podium
- Post-SMOTE training split: 50% / 50% (balanced)
- Test set: **untouched** — retained original 77.71/22.29 distribution for honest evaluation

### Model Development
**Five candidate classifiers benchmarked:**

| Model | Key Strength | Limitation |
|---|---|---|
| Logistic Regression | Interpretable, baseline | Assumes linear decision boundary |
| Decision Tree | Rule-based, explainable | High variance, overfits |
| Random Forest | Robust ensemble, low variance | Less tunable than boosting |
| **XGBoost** ✅ | **Best F1 + AUC, gradient boosting** | **Requires hyperparameter tuning** |
| LightGBM | Fastest training | Marginally lower F1 on this dataset |

**Optimization:** GridSearchCV with 5-fold stratified cross-validation

```python
param_grid = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [3, 5, 7],
    'learning_rate':    [0.05, 0.1, 0.2],
    'subsample':        [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
```

**Best parameters selected:** `n_estimators=300`, `max_depth=7`, `learning_rate=0.2`, `subsample=1.0`

**Optimization target:** F1-Score (Podium class), not Accuracy — chosen to penalize false negatives appropriately in the context of class imbalance.

### Evaluation Metrics Framework

**Why NOT Accuracy Alone:**
- A naive classifier predicting "Non-Podium" for every entry achieves 77.7% accuracy while detecting 0% of actual podiums
- **Primary metric: ROC-AUC** (discriminative power across all thresholds)
- **Secondary metric: Podium F1-Score** (harmonic mean of precision and recall on minority class)

| Metric | Formula | Why It Matters |
|---|---|---|
| **ROC-AUC** | Area under ROC curve | Overall discriminative power, threshold-independent |
| **Recall (Podium)** | TP / (TP + FN) | Minimizes missed podium predictions |
| **Precision (Podium)** | TP / (TP + FP) | Controls false alarm rate |
| **F1-Score (Podium)** | 2 × (P × R) / (P + R) | Balanced measure for imbalanced classes |

---

## 📈 Model Performance

### Final Model: XGBoost — Test Set Results (n = 3,137)

| Metric | Non-Podium | Podium |
|---|---|---|
| **Precision** | 0.9650 | **0.8575** |
| **Recall** | 0.9582 | **0.8784** |
| **F1-Score** | 0.9616 | **0.8678** |
| **Support** | 2,438 | 699 |

**Overall Metrics:**
- **Accuracy:** 94.04%
- **Weighted F1-Score:** 0.94
- **ROC-AUC: 0.9738** ← *Primary evaluation metric*

### Confusion Matrix

| | Predicted: Non-Podium | Predicted: Podium |
|---|---|---|
| **Actual: Non-Podium** | 2,336 ✅ (TN) | 102 ⚠️ (FP) |
| **Actual: Podium** | 85 ❌ (FN) | 614 ✅ (TP) |

- **614 of 699** true podium finishes correctly identified (**87.8% recall**)
- **Only 85 podiums missed** (false negatives) across the entire test set
- **102 false positives** — acceptable false alarm rate given the high recall achieved

### Prediction Confidence Distribution (Full Dataset: 15,681 entries)

| Confidence Level | Entries | Definition |
|---|---|---|
| **High Confidence** | 13,618 (86.9%) | Probability < 0.25 or > 0.75 |
| **Moderate** | 1,371 (8.7%) | Probability 0.25–0.40 or 0.60–0.75 |
| **Uncertain** | 692 (4.4%) | Probability 0.40–0.60 (flag for review) |

### Feature Importance (Numeric Features)

| Feature | Importance Score | Interpretation |
|---|---|---|
| `grid` | 0.0379 | Strongest single numeric predictor — confirms qualifying dominance |
| `qualifying_delta` | 0.0070 | Pace gap from pole — signals relative competitiveness |
| `driver_momentum` | 0.0055 | Recent form — captures in-season confidence and car development |
| `driver_age` | 0.0036 | Career stage effects — prime-age drivers slightly elevated |

> **Note:** Constructor and circuit dummy variables occupy the highest importance slots individually (Ferrari: 0.0155, etc.), but their collective contribution is spread across 237 binary features.

---

## 📁 Project Structure

```
F1-Podium-Prediction/
├── README.md                          # This file
├── F1-Analysis.ipynb                  # Main analysis notebook (Chapters 1–5)
├── model/
│   └── xgboost_f1_model.pkl          # Trained XGBoost classifier
├── dataset/
│   ├── f1_circuits.csv               # Circuit metadata
│   ├── f1_constructor_standings.csv  # Season-level constructor standings
│   ├── f1_constructors.csv           # Constructor reference table
│   ├── f1_driver_standings.csv       # Season-level driver standings
│   ├── f1_drivers.csv                # Driver reference table
│   ├── f1_qualifying.csv             # Qualifying session times (Q1/Q2/Q3)
│   ├── f1_races.csv                  # Race schedule and metadata
│   ├── f1_results.csv                # Race-by-race finishing results
│   ├── f1_master_dataset.csv         # Merged & cleaned dataset (model input)
│   └── f1_predictions.csv            # Model output with podium_probability scores
└── dashboard/
    └── [Tableau workbook — in progress]
```

---

## 🚀 Installation & Setup

### Requirements
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required Libraries:

```
pandas >= 1.3.0
numpy >= 1.20.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
xgboost >= 1.5.0
imbalanced-learn >= 0.8.0
shap >= 0.40.0
joblib >= 1.0.0
```

### Installation Steps
```bash
# Clone repository
git clone https://github.com/yourusername/f1-podium-prediction.git
cd f1-podium-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Quick Start
```python
# Open F1-Analysis.ipynb and run cells sequentially:
# Chapter 1: Business Problem Understanding
# Chapter 2: Data Understanding & Preparation (8-table merge + cleaning)
# Chapter 3: Exploratory Data Analysis
# Chapter 4: Machine Learning Modeling (XGBoost + SHAP)
# Chapter 5: Conclusion & Recommendations

# Or load the pre-trained model directly:
import joblib, pandas as pd

model = joblib.load('model/xgboost_f1_model.pkl')
df    = pd.read_csv('dataset/f1_predictions.csv')

# Filter to a specific race and view predictions
race_view = df[(df['season'] == 2025) & (df['race_name'] == 'Bahrain Grand Prix')]
print(race_view[['driver_name', 'grid', 'podium_probability', 'uncertainty_flag']]
      .sort_values('podium_probability', ascending=False))
```

---

## 📊 Analysis Workflow

### Chapter 1: Business Problem Understanding
✅ **Completed**
- Business context defined: F1 podium prediction as a sports intelligence and strategy problem
- Analytical objectives articulated (binary classification, feature insight, temporal analysis)
- Evaluation metric framework established (ROC-AUC primary, F1-Score secondary)
- SMOTE justification: 22.29% podium rate requires class imbalance correction

### Chapter 2: Data Understanding & Preparation
✅ **Completed**
- 8 source CSV files loaded and merged via multi-key relational joins
- Qualifying time strings parsed to milliseconds
- Q3 → Q2 → Q1 fallback strategy implemented for `qualifying_delta`
- Low-information columns dropped (`time`, `fastest_lap`, `fastest_lap_rank`)
- `driver_momentum` engineered as 3-race rolling points sum
- `driver_age` calculated from date of birth to race date
- Final master dataset: 15,681 rows × 21 columns, zero missing values

### Chapter 3: Exploratory Data Analysis
✅ **Completed**
- Grid position vs podium rate: non-linear analysis (84.1% at Grid 1 → 1.4% at Grid 20)
- Constructor all-time podium leaderboard (Ferrari: 864, McLaren: 543)
- Circuit-specific constructor heatmap (top 8 teams × top 15 circuits)
- Era-segmented analysis: Classic vs Pre-Q3 vs Modern podium rate trends
- Driver momentum distribution and podium correlation
- Nationality breakdown of all-time podium finishes

### Chapter 4: Machine Learning Modeling
✅ **Completed**
- 5-model benchmark comparison (Logistic Regression → LightGBM)
- XGBoost selected as final model
- SMOTE applied to training set (50/50 balance)
- GridSearchCV hyperparameter tuning (5-fold stratified CV)
- SHAP explainability: global feature importance + individual prediction breakdown
- Final evaluation on held-out test set (n=3,137): ROC-AUC 0.9738, Podium F1 0.8678
- `f1_predictions.csv` exported with `podium_probability`, `podium_prediction`, `uncertainty_flag`

### Chapter 5: Conclusion & Recommendations
✅ **Completed**
- Technical evaluation summary with full metrics documentation
- 5 data-driven recommendations for teams and analysts
- 5 modeling limitations with mitigation strategies
- Tableau dashboard architecture (4-page design) and data export plan

### Tableau Dashboard
🔄 **In Progress**
- 11 chart sheets across 4 dashboard pages
- Page 1: F1 Historical Overview (75-year macro trends)
- Page 2: Driver & Constructor Deep Dive (heatmap, scatter, career analysis)
- Page 3: Podium Prediction — ML Output (live probability table, confidence histogram)
- Page 4: Model Monitoring & Validation (accuracy by season, error analysis)

---

## 🎯 Key Insights & Recommendations

### Insight 1: Qualifying Position is the Single Most Predictive Factor
**Finding:** Grid 1 drivers achieve an 84.1% historical podium rate — more than twice the probability of Grid 4, and 60× higher than Grid 20.

**Recommendation:**
- Teams should treat Saturday qualifying as equally critical as Sunday race preparation
- Investment in one-lap aerodynamic balance, tire warm-up procedures, and low-fuel setup directly translates to higher expected podium probability before a wheel is turned in the race
- Strategists should anchor race strategy planning to grid position as the primary input variable

---

### Insight 2: Constructor Identity Creates Persistent Podium Probability Windows
**Finding:** Constructor-specific features dominate the model's top importance slots. Ferrari's 864 all-time podiums and circuit-specific team performance patterns (e.g., Mercedes at Silverstone, Red Bull at Bahrain) create structural probability advantages that persist across driver changes.

**Recommendation:**
- Race analysts should maintain a circuit-level constructor performance matrix updated after each event
- Fantasy F1 players should prioritize constructor track record at the specific circuit over season-aggregate standings
- Teams entering new circuits should study historical performance of their closest technical analogs (similar aerodynamic philosophy, power unit supplier)

---

### Insight 3: Driver Momentum Outperforms Season Standings for Short-Term Prediction
**Finding:** `driver_momentum` (3-race rolling points) adds predictive signal beyond season-aggregate standings. A driver with 25+ points in the last 3 races has measurably higher podium probability than their championship position alone suggests.

**Recommendation:**
- Team strategists and media analysts should track 3-race rolling form as the primary form indicator
- Fantasy F1 team selectors should weight recent momentum heavily over season-long averages, especially after car development updates mid-season
- Drivers in negative momentum windows may benefit from lower-risk race strategies to accumulate consistent points rather than chasing unlikely wins

---

### Insight 4: The Modern Era (2006+) is Significantly More Predictable
**Finding:** Podium rate concentration has increased in the modern era — dominant constructors win more consistently, and the qualifying → race outcome correlation has strengthened with the current aerodynamic regulations.

**Recommendation:**
- Predictive models trained on modern-era data (2006+) will outperform full-history models for current race predictions
- The model should ideally be deployed with a **modern-era submodel** for active race forecasting, while the full-history model serves historical analysis purposes

---

### Insight 5: Uncertainty Flagging is Operationally Critical
**Finding:** 4.4% of predictions (692 entries) fall in the 0.40–0.60 probability range — genuinely uncertain predictions where the model's confidence is lowest. These align with non-standard race scenarios (safety cars, weather, penalties).

**Recommendation:**
- Dashboard users should apply heightened scrutiny to "Uncertain" flagged predictions
- These cases are candidates for human expert override based on weekend-specific context (weather, penalties, strategy calls) not captured in static historical features
- Do not assign binary labels to uncertain predictions in operational contexts

---

## 📈 Expected Business Impact

### Model Performance Summary

| Metric | Value | Benchmark |
|---|---|---|
| **ROC-AUC** | **0.9738** | > 0.95 ✅ Exceeded |
| **Podium F1-Score** | **0.8678** | > 0.80 ✅ Exceeded |
| **Recall (Podium)** | **87.8%** | > 85% ✅ Exceeded |
| **Accuracy** | **94.04%** | Reference only |
| **Correct Predictions (full dataset)** | **14,795 / 15,681** | **94.35%** |

### Prediction Output for Analysts

| Metric | Value |
|---|---|
| **True Podiums Correctly Identified (test set)** | 614 of 699 |
| **Podiums Missed (false negatives)** | 85 |
| **False Alarms (false positives)** | 102 |
| **High Confidence Predictions** | 13,618 (86.9%) |
| **Uncertain Predictions (flag for review)** | 692 (4.4%) |

---

## 👥 Stakeholders & Use Cases

### Primary Users
1. **Race Analysts & Broadcasters**
   - Use pre-race probability table to anchor commentary and analysis
   - Identify high-uncertainty predictions for narrative focus (upset potential)

2. **Team Strategists**
   - Quantify podium probability before committing to aggressive vs conservative race strategies
   - Benchmark opponent podium probability by grid slot

3. **Fantasy F1 Players**
   - Objective podium probability score per driver per race
   - Uncertainty flags to identify high-risk/high-reward selections

4. **Sports Data Scientists**
   - Reproducible ML pipeline across 75 years of normalized F1 data
   - Baseline model for further development (multi-class P1/P2–P3/Points, live telemetry integration)

---

## 📚 References & Data Sources

- **Ergast Developer API:** [ergast.com/mrd](http://ergast.com/mrd/) — Primary data source for all race, qualifying, and standings data
- **Formula 1 Official:** [formula1.com](https://www.formula1.com) — Historical race records reference
- **SHAP Library:** Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
- **Imbalanced-learn (SMOTE):** Chawla, N. V., et al. (2002). *SMOTE: Synthetic minority over-sampling technique.* Journal of Artificial Intelligence Research.
- **XGBoost:** Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD.

---

## 📊 Tableau Interactive Dashboard

🔄 **Dashboard currently in development — link will be updated upon completion.**

*4-page interactive dashboard covering:*
- *Page 1: F1 Historical Overview (1950–2026)*
- *Page 2: Driver & Constructor Deep Dive*
- *Page 3: ML Podium Prediction Output (live probability table)*
- *Page 4: Model Monitoring & Validation*

---

## 📝 Author Notes

This project demonstrates the application of end-to-end data science methodology to one of the world's most statistically rich sports datasets. The approach prioritizes:

- **Integration Before Analysis:** Eight normalized source tables were merged into a single analytically coherent master dataset before any modeling began
- **Domain-Aligned Feature Engineering:** `qualifying_delta`, `driver_momentum`, and `driver_age` were designed around actual F1 domain knowledge, not generic ML defaults
- **Honest Evaluation:** SMOTE was applied only to the training set; the test set retained its original imbalanced distribution for realistic performance measurement
- **Explainability First:** SHAP values ensure every prediction can be explained in terms a non-technical stakeholder can understand
- **Actionability Over Complexity:** Every analytical finding connects to a concrete recommendation for teams, analysts, or strategists

---

## 📄 License

This project is developed as a personal data science portfolio project — 2026.

All Rights Reserved.

---

## 💬 Questions?

For technical questions regarding the ML pipeline, data cleaning methodology, or feature engineering decisions, refer to the detailed inline documentation in `F1-Analysis.ipynb`.

For dashboard-related questions, refer to the Tableau guide included in the project documentation.

**Last Updated:** May 2026  
**Status:** Analysis & Modeling Complete | Dashboard In Progress
