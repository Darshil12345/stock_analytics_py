"""
models/model_registry.py — Centralised sklearn estimator catalogue.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier

MODEL_MAP: dict = {
    "Logistic Regression":   LogisticRegression(max_iter=2000),
    "Decision Tree":         DecisionTreeClassifier(max_depth=5, random_state=42),
    "Gaussian Naive Bayes":  GaussianNB(),
    "SVC (RBF)":             SVC(kernel="rbf", probability=True, random_state=42),
    "KNN (k=5)":             KNeighborsClassifier(n_neighbors=5),
    "Random Forest":         RandomForestClassifier(n_estimators=100, max_depth=10,
                                                     random_state=42, n_jobs=-1),
    "Gradient Boosting":     GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost":              AdaBoostClassifier(n_estimators=100, random_state=42),
}

# Models whose SHAP can use the fast TreeExplainer
TREE_MODELS = {"Random Forest", "Decision Tree", "Gradient Boosting", "AdaBoost"}
