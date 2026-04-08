"""
helpers.py — Reusable pipeline helpers for the Feature Engineering Capstone.
StaySmart Hotels | Data Preprocessing Assignment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              confusion_matrix, ConfusionMatrixDisplay,
                              classification_report)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────

def load_hotel_data(url: str) -> pd.DataFrame:
    """Load and return the hotel bookings CSV from a URL."""
    df = pd.read_csv(url)
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────
# 2. BASELINE MODEL HELPERS
# ─────────────────────────────────────────────────────────

def build_baseline_pipeline(num_cols, cat_cols):
    """Build a minimal preprocessing + LogisticRegression baseline pipeline."""
    from sklearn.linear_model import LogisticRegression

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    return model


def evaluate_model(model, X_test, y_test, label="Model"):
    """Print and return evaluation metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Not Cancelled", "Cancelled"]))
    return {"label": label, "accuracy": acc, "roc_auc": auc, "f1": f1}


def plot_confusion_matrix(model, X_test, y_test, title="Confusion Matrix", save_path=None):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Cancelled", "Cancelled"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────
# 3. CURSE OF DIMENSIONALITY
# ─────────────────────────────────────────────────────────

def curse_of_dimensionality_demo(dims=(2, 10, 50, 200), n_samples=500, save_path=None):
    """Plot pairwise distance distributions for increasing dimensions."""
    from scipy.spatial.distance import pdist
    fig, axes = plt.subplots(1, len(dims), figsize=(16, 4), sharey=False)
    fig.suptitle("Curse of Dimensionality — Pairwise Euclidean Distance Distributions",
                 fontsize=13, fontweight='bold')
    for ax, d in zip(axes, dims):
        X, _ = make_classification(n_samples=n_samples, n_features=d,
                                   n_informative=max(2, d//2), n_redundant=0,
                                   random_state=42)
        dists = pdist(X, metric='euclidean')
        ax.hist(dists, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
        ax.set_title(f"{d} features\nmin={dists.min():.1f}, max={dists.max():.1f}", fontsize=10)
        ax.set_xlabel("Euclidean Distance")
        ax.set_ylabel("Frequency")
        ratio = dists.max() / (dists.min() + 1e-9)
        ax.text(0.95, 0.95, f"ratio={ratio:.1f}", ha='right', va='top',
                transform=ax.transAxes, fontsize=8, color='darkred')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    print("Curse of dimensionality plot saved.")


def nn_distance_ratio_plot(dims=(2, 10, 50, 200), n_samples=300, save_path=None):
    """Plot nearest/farthest neighbor distance ratio."""
    from sklearn.metrics import pairwise_distances
    ratios = []
    for d in dims:
        X, _ = make_classification(n_samples=n_samples, n_features=d,
                                   n_informative=max(2, d//2), n_redundant=0,
                                   random_state=42)
        D = pairwise_distances(X)
        np.fill_diagonal(D, np.inf)
        min_d = D.min(axis=1)
        np.fill_diagonal(D, 0)
        max_d = D.max(axis=1)
        ratios.append(np.mean(min_d / (max_d + 1e-9)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dims, ratios, marker='o', color='crimson', linewidth=2)
    ax.set_xlabel("Number of Features (Dimensions)")
    ax.set_ylabel("Mean NN Distance Ratio (min/max)")
    ax.set_title("Nearest Neighbor Distance Ratio vs Dimensionality")
    ax.set_xscale("log")
    for x, y in zip(dims, ratios):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────
# 4. NUMERIC PREPROCESSING
# ─────────────────────────────────────────────────────────

def plot_before_after_scaling(df, cols, save_path=None):
    """Box plots comparing raw vs scaled versions."""
    scalers = {
        "Original": None,
        "MinMaxScaler": MinMaxScaler(),
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler()
    }
    n = len(cols)
    fig, axes = plt.subplots(len(scalers), n, figsize=(n * 3, len(scalers) * 2.5))
    fig.suptitle("Before vs After Scaling (Box Plots)", fontsize=13, fontweight='bold')
    for row_idx, (name, scaler) in enumerate(scalers.items()):
        data = df[cols].copy()
        data = data.fillna(data.median())
        if scaler:
            data = pd.DataFrame(scaler.fit_transform(data), columns=cols)
        for col_idx, col in enumerate(cols):
            ax = axes[row_idx, col_idx]
            ax.boxplot(data[col], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
            ax.set_title(f"{col}\n({name})", fontsize=8)
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def scaling_summary_stats(df, cols):
    """Return summary stats for original and each scaler."""
    results = {}
    data = df[cols].fillna(df[cols].median())
    for name, scaler in [("Original", None),
                          ("MinMax", MinMaxScaler()),
                          ("Standard", StandardScaler()),
                          ("Robust", RobustScaler())]:
        d = pd.DataFrame(scaler.fit_transform(data), columns=cols) if scaler else data
        results[name] = d.agg(['mean', 'std']).round(3)
    return results


# ─────────────────────────────────────────────────────────
# 5. DISTANCE METRICS EXPERIMENT
# ─────────────────────────────────────────────────────────

def knn_scaling_experiment(X_train, X_test, y_train, y_test, k=15):
    """Run KNN with no scaling, StandardScaler, RobustScaler; Euclidean + Manhattan."""
    results = []
    configs = [
        ("No Scaling",      None,           "euclidean"),
        ("No Scaling",      None,           "manhattan"),
        ("StandardScaler",  StandardScaler(), "euclidean"),
        ("StandardScaler",  StandardScaler(), "manhattan"),
        ("RobustScaler",    RobustScaler(),   "euclidean"),
        ("RobustScaler",    RobustScaler(),   "manhattan"),
    ]
    for scaler_name, scaler, metric in configs:
        Xtr = scaler.fit_transform(X_train) if scaler else X_train
        Xte = scaler.transform(X_test) if scaler else X_test
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(Xtr, y_train)
        y_pred = knn.predict(Xte)
        y_prob = knn.predict_proba(Xte)[:, 1]
        results.append({
            "Scaler": scaler_name,
            "Distance": metric,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────
# 6. DATE/TIME FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────

def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from hotel bookings dataset."""
    df = df.copy()
    month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
                 'May': 5, 'June': 6, 'July': 7, 'August': 8,
                 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    df['arrival_month_num'] = df['arrival_date_month'].map(month_map)

    # Season
    def month_to_season(m):
        if m in [12, 1, 2]:  return 'Winter'
        if m in [3, 4, 5]:   return 'Spring'
        if m in [6, 7, 8]:   return 'Summer'
        return 'Autumn'
    df['season'] = df['arrival_month_num'].apply(month_to_season)

    # Quarter
    df['quarter'] = ((df['arrival_month_num'] - 1) // 3 + 1).astype(str).apply(lambda x: f"Q{x}")

    # Weekend arrival
    df['is_weekend_arrival'] = df['arrival_date_day_of_month'].apply(
        lambda d: 1 if d % 7 in [5, 6] else 0)

    # Lead time buckets
    df['lead_time_bucket'] = pd.cut(df['lead_time'],
                                     bins=[-1, 7, 30, 90, 180, 500, 10000],
                                     labels=['Same week', '1-4 weeks', '1-3 months',
                                             '3-6 months', '6-18 months', '18+ months'])
    # Year
    df['arrival_year'] = df['arrival_date_year']
    return df


# ─────────────────────────────────────────────────────────
# 7. FEATURE CONSTRUCTION
# ─────────────────────────────────────────────────────────

def construct_features(df: pd.DataFrame, train_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Construct domain-driven features.
    train_df is used for group aggregations to AVOID leakage.
    """
    df = df.copy()

    # Total guests (avoid div-by-zero)
    df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies']
    df['total_guests'] = df['total_guests'].replace(0, 1)

    # Ratio features
    df['price_per_person']  = df['adr'] / df['total_guests']
    df['special_req_rate']  = df['total_of_special_requests'] / (
        df['stays_in_weekend_nights'] + df['stays_in_week_nights'] + 1)

    # Total nights
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['total_nights'] = df['total_nights'].replace(0, 1)

    # Revenue proxy
    df['total_revenue'] = df['adr'] * df['total_nights']

    # Interaction features
    df['adr_x_lead_time']     = df['adr'] * df['lead_time']
    df['nights_x_guests']     = df['total_nights'] * df['total_guests']

    # Binary / flag features
    df['is_family']           = ((df['children'].fillna(0) + df['babies']) > 0).astype(int)
    df['is_repeated_guest']   = df['is_repeated_guest'].astype(int)
    df['has_special_request'] = (df['total_of_special_requests'] > 0).astype(int)
    df['has_deposit']         = (df['deposit_type'] != 'No Deposit').astype(int)

    # Binarization: high-value booking
    adr_thresh = df['adr'].quantile(0.75)
    df['is_high_value_booking'] = (df['adr'] > adr_thresh).astype(int)

    # Group aggregation: avg adr by hotel type (leakage-safe: train only)
    ref = train_df if train_df is not None else df
    hotel_avg_adr = ref.groupby('hotel')['adr'].mean().rename('hotel_avg_adr')
    df = df.join(hotel_avg_adr, on='hotel')

    country_cancel_rate = ref.groupby('country')['is_canceled'].mean().rename('country_cancel_rate')
    df = df.join(country_cancel_rate, on='country')
    df['country_cancel_rate'] = df['country_cancel_rate'].fillna(ref['is_canceled'].mean())

    return df


# ─────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────

def plot_feature_importance(importances: pd.Series, title: str, top_n=15, save_path=None):
    top = importances.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def correlation_filter(X: pd.DataFrame, threshold=0.85) -> list:
    """Return list of columns to DROP due to high correlation."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return to_drop
