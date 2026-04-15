import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import KNNImputer, SimpleImputer


def load_and_merge_data(expr_path, meth_path):
    print("🔄 [Step 1] 正在加载多组学数据...")

    if not os.path.exists(expr_path) or not os.path.exists(meth_path):
        raise FileNotFoundError(f"找不到文件，请检查路径: {expr_path} 或 {meth_path}")

    df_expr = pd.read_csv(expr_path, index_col=0)
    df_meth = pd.read_csv(meth_path, index_col=0)

    expr_cols_map = {c: c.replace('-', '').replace(' ', '') for c in df_expr.columns}
    meth_cols_map = {c: c.replace('-', '').replace(' ', '') for c in df_meth.columns}

    df_expr.rename(columns=expr_cols_map, inplace=True)
    df_meth.rename(columns=meth_cols_map, inplace=True)

    common_samples = list(set(df_expr.columns) & set(df_meth.columns))
    common_samples.sort()

    if len(common_samples) == 0:
        raise ValueError("❌ 错误：两个数据集没有重合的样本名！")

    print(f"   - 发现 {len(common_samples)} 个共有样本，正在对齐...")

    X_expr = df_expr[common_samples].T
    X_meth = df_meth[common_samples].T

    labels = []
    for sample in common_samples:
        if "NC" in sample:
            labels.append(0)
        elif "LN" in sample:
            labels.append(2)
        else:
            labels.append(1)

    return X_expr, X_meth, np.array(labels), common_samples


def feature_selection(X, y, n_features=2000, omic_name="Omic"):
    print(f"   - [{omic_name}] 初步筛选 (Top {n_features})...")

    X = X.dropna(axis=1, how='all')

    if X.isnull().values.any():
        if X.shape[1] > 10000:
            imputer = SimpleImputer(strategy='mean')
        else:
            imputer = KNNImputer(n_neighbors=5)
        X_values = imputer.fit_transform(X)
    else:
        X_values = X.values

    selector = SelectKBest(score_func=f_classif, k=n_features)
    selector.fit(X_values, y)
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]

    return X[selected_features]