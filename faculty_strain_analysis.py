import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
import xgboost as xgb

# Load your data
df = pd.read_csv('FacultyStrain.csv')

# Assume StressLevel is the target
target = 'StressLevel'
features = [col for col in df.columns if col != target]

# Standardize numeric features
X = df[features]
y = df[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- Figure 1: Correlation Heatmap ----------
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Figure 1. Correlation Heatmap among Faculty Stress Variables')
plt.tight_layout()
plt.savefig('figure1_correlation_heatmap.png')
plt.close()

# ---------- Figure 2: Boxplot of Stress vs Sleep ----------
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['StressLevel'], y=df['SleepHours'])
plt.title('Figure 2. Stress Level vs. Sleep Hours')
plt.xlabel('Stress Level')
plt.ylabel('Sleep Hours')
plt.tight_layout()
plt.savefig('figure2_boxplot_stress_sleep.png')
plt.close()

# ---------- Figure 3: K-Means Clustering ----------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('Figure 3. K-Means Clustering of Faculty Stress Profiles')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.tight_layout()
plt.savefig('figure3_kmeans_clustering.png')
plt.close()

# ---------- Figure 4: Feature Importance (Random Forest) ----------
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', figsize=(10, 6))
plt.title('Figure 4. Feature Importance from Random Forest')
plt.tight_layout()
plt.savefig('figure4_rf_feature_importance.png')
plt.close()

# ---------- Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------- Figure 5: Confusion Matrix (XGBoost) ----------
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Figure 5. Confusion Matrix (XGBoost)')
plt.tight_layout()
plt.savefig('figure5_confusion_matrix_xgb.png')
plt.close()

# ---------- Figure 6: ROC Curves ----------
plt.figure(figsize=(8, 6))
for cls in sorted(y.unique()):
    y_bin = (y_test == cls).astype(int)
    y_score = xgb_model.predict_proba(X_test)[:, cls - 1]
    fpr, tpr, _ = roc_curve(y_bin, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('Figure 6. ROC Curves for Stress Level Prediction')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figure6_roc_curve.png')
plt.close()
