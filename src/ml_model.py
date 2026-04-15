from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_ml_classification(X, y, output_dir):
    print("\n🤖 [Step 3] 运行机器学习验证 (优化版 Pipeline)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=50)),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            class_weight='balanced',
            random_state=42
        ))
    ])

    cv = StratifiedKFold(n_splits=5)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"   - 5折交叉验证准确率 (Train): {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n   - 独立测试集报告:")
    target_names = ['Normal', 'SLE', 'SLE+LN']
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix (Optimized)')
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"✅ 混淆矩阵保存至: {save_path}")

    return pipeline