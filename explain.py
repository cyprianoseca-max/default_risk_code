# utils/explain.py

import numpy as np
import pandas as pd

def build_feature_importance_table(feature_names, importance_values):
    idx = np.argsort(importance_values)[::-1]
    sorted_names = np.array(feature_names)[idx]
    sorted_values = np.array(importance_values)[idx]

    df = pd.DataFrame({
        "特征名称": sorted_names,
        "重要性值": sorted_values,
        "重要性排序": range(1, len(sorted_names) + 1)
    })
    return df