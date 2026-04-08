StaySmart Hotels — Feature Engineering Capstone
Graded Assignment 1 | Data Preprocessing & Feature Engineering
Overview
Predict hotel booking cancellations (`is_canceled`) using feature engineering and preprocessing techniques.
Dataset: Hotel Bookings — swapnilsaurav/Dataset on GitHub
Repository Structure
```
FeatureEngineering_Capstone/
├── FeatureEngineering_Capstone.ipynb   # Main notebook (all 8 tasks)
├── src/
│   └── helpers.py                      # Reusable helper functions & pipelines
├── report/
│   └── Report.pdf                      # Full written report with graphs
├── requirements.txt
└── README.md
```
Setup & Run
Option 1 — Google Colab (Recommended)
Upload the notebook to Google Colab
Run: `!pip install -r requirements.txt`
Run all cells
Option 2 — Local Setup
```bash
# Clone or unzip the repository
cd FeatureEngineering_Capstone

# Create virtual environment
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook FeatureEngineering_Capstone.ipynb
```
Tasks Covered
Task	Topic	Key Output
1	Baseline Model	Accuracy 0.52, ROC-AUC 0.53
2	Curse of Dimensionality	Distance distribution + NN ratio plots
3	Numeric Preprocessing	Binning, Binarization, Scaler comparison
4	Distance Metrics & Scaling	KNN × 6 configs (3 scalers × 2 metrics)
5	End-to-End Pipeline	Modular ColumnTransformer + CV
6	Feature Extraction	6 date/time features + encoding
7	Feature Construction	10+ engineered features + leakage prevention
8	Feature Importance	RF importance + Mutual Information + selection
Final	Comparison Table	Before vs After summary
Key Findings
Best single predictor: `deposit_type` (non-refundable bookings rarely cancel)
Best engineered feature: `country_cancel_rate` (group aggregation, train-only)
Best scaler: `RobustScaler` (handles skewed adr/lead_time distributions)
Performance gain: Accuracy from 52.3% → 64.8% through feature engineering alone
Notes on Leakage Prevention
All group aggregations (country_cancel_rate, hotel_avg_adr) are computed on training data only and applied to test/validation sets. All transformations are inside scikit-learn Pipelines that only fit on training data.
