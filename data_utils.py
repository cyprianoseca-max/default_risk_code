# utils/data_utils.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)

def generate_credit_data(n=5000, seed=42):
    np.random.seed(seed)

    income = np.random.lognormal(mean=10, sigma=0.5, size=n)
    debt_ratio = np.random.beta(a=2, b=5, size=n)
    credit_history = np.random.randint(1, 20, size=n)
    age = np.random.randint(20, 60, size=n)
    recent_inquiries = np.random.poisson(lam=2, size=n)
    utilization = np.clip(np.random.normal(0.5, 0.2, size=n), 0, 1)
    overdue_count = np.random.poisson(lam=0.5, size=n)

    risk_score = (
        0.35 * debt_ratio +
        0.25 * utilization +
        0.20 * overdue_count +
        0.10 * recent_inquiries -
        0.000001 * income -
        0.02 * credit_history -
        0.01 * age +
        np.random.normal(0, 0.1, size=n)
    )

    threshold = np.quantile(risk_score, 0.85)
    default = (risk_score >= threshold).astype(int)

    df = pd.DataFrame({
        "Income": income,
        "Debt_Ratio": debt_ratio,
        "Credit_History": credit_history,
        "Age": age,
        "Recent_Inquiries": recent_inquiries,
        "Utilization": utilization,
        "Overdue_Count": overdue_count,
        "Default": default
    })
    return df

def split_data(df):
    X = df.drop(columns=["Default"])
    y = df["Default"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def standardize_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler