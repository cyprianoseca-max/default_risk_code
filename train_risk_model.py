# experiments/train_risk_model.py

import os
import json
import numpy as np
import pandas as pd

from config import CONFIG
from utils.data_utils import ensure_dirs, generate_credit_data, split_data, standardize_data
from utils.metrics import evaluate_binary_classifier
from utils.plotting import plot_feature_importances, plot_default_distribution, plot_feature_distributions, plot_roc_curves
from models.risk_models import LRModel, RFModel
from models.tabnet_model import TabNetRiskModel
from sklearn.metrics import roc_curve

def run_risk_model_experiment():
    ensure_dirs()

    df = generate_credit_data(n=CONFIG["n_samples"], seed=CONFIG["random_seed"])
    df.to_csv(CONFIG["raw_data_path"], index=False)

    # 绘制 图 4-2 和 图 4-3
    plot_default_distribution(df, os.path.join(CONFIG["results_path"], "default_distribution.png"))
    plot_feature_distributions(df, os.path.join(CONFIG["results_path"], "feature_distributions.png"))

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_s, X_val_s, X_test_s, scaler = standardize_data(X_train, X_val, X_test)

    results = {}
    roc_data = {}

    # 逻辑回归
    lr_model = LRModel(max_iter=CONFIG["lr_max_iter"])
    lr_model.fit(X_train_s, y_train)
    lr_pred = lr_model.predict(X_test_s)
    lr_prob = lr_model.predict_proba(X_test_s)
    results["LogisticRegression"] = evaluate_binary_classifier(y_test, lr_pred, lr_prob)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
    roc_data["Logistic Regression"] = (fpr_lr, tpr_lr, results["LogisticRegression"]["AUC"])

    # 随机森林
    rf_model = RFModel(n_estimators=CONFIG["rf_n_estimators"], random_state=CONFIG["random_seed"])
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)
    results["RandomForest"] = evaluate_binary_classifier(y_test, rf_pred, rf_prob)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
    roc_data["Random Forest"] = (fpr_rf, tpr_rf, results["RandomForest"]["AUC"])

    # TabNet
    tabnet_model = TabNetRiskModel(seed=CONFIG["random_seed"])
    tabnet_model.fit(
        X_train.values, y_train.values,
        X_val.values, y_val.values,
        max_epochs=CONFIG["tabnet_epochs"],
        batch_size=CONFIG["tabnet_batch_size"]
    )
    tab_pred = tabnet_model.predict(X_test.values)
    tab_prob = tabnet_model.predict_proba(X_test.values)
    results["TabNet"] = evaluate_binary_classifier(y_test, tab_pred, tab_prob)
    fpr_tab, tpr_tab, _ = roc_curve(y_test, tab_prob)
    roc_data["TabNet"] = (fpr_tab, tpr_tab, results["TabNet"]["AUC"])

    # 绘制 图 5-1 ROC曲线
    plot_roc_curves(roc_data, os.path.join(CONFIG["results_path"], "roc_curves_comparison.png"))

    print("风险模型评估结果：")
    for k, v in results.items():
        print(k, v)

    with open(os.path.join(CONFIG["results_path"], "risk_model_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 绘制并保存特征重要性为 png 格式
    feature_names = X_train.columns.tolist()
    importances = tabnet_model.feature_importances()
    plot_feature_importances(
        feature_names, 
        importances, 
        os.path.join(CONFIG["results_path"], "tabnet_feature_importances.png")
    )
    print("TabNet 特征重要性图表已保存。")