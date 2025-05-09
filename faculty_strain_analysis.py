import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# Load the dataset (update the path if needed)
df = pd.read_csv("FacultyStrain.csv")

# OPTIONAL: View column names
print("Columns:", df.columns.tolist())

# Encode categorical variables
le = LabelEncoder()
for col in ['Gender', 'Age Group', 'Academic Rank']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# ==============================
# Figure 1: Correlation Heatmap
# ==============================
stress_vars = df.select_dtypes(include=['int64', 'float64']).iloc[:, 3:]  # stress-related columns
plt.figure(figsize=(10, 8))
sns.heatmap(stress_vars.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Figure 1: Correlation Heatmap of Stress and Impact Variables")
plt.tight_layout()
plt.savefig("figure1_correlation_heatmap.png")
plt.close()

# ==========================================
# Figure 2: K-Means Clustering (Sleep vs Workload)
# ==========================================
cols_needed = ['Sleep Hours per Night', 'Stress due to Teaching Workload']
if all(col in df.columns for col in cols_needed):
    cluster_df = df[cols_needed].dropna()
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_df)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(cluster_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=cluster_df[cols_needed[0]], y=cluster_df[cols_needed[1]], hue=labels, palette='Set2')
    plt.title("Figure 2: K-Means Clustering Based on Sleep and Workload")
    plt.xlabel(cols_needed[0])
    plt.ylabel(cols_needed[1])
    plt.tight_layout()
    plt.savefig("figure2_kmeans_clustering.png")
    plt.close()

# ====================================================
# Figure 3: Feature Importance from Random Forest Model
# ====================================================
target_col = 'Impact of Stress on Physical Health'
if target_col in df.columns:
    X_rf = df.drop(columns=[target_col]).dropna()
    y_rf = (df[target_col] >= 4).astype(int)  # binary classification: high stress = 1

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_rf, y_rf)

    importances = pd.Series(rf_model.feature_importances_, index=X_rf.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values[:10], y=importances.index[:10], palette='crest')
    plt.title("Figure 3: Feature Importance - Predicting High Health Impact")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("figure3_feature_importance.png")
    plt.close()

print("✅ Figures generated and saved as PNG files:")
print(" - figure1_correlation_heatmap.png")
print(" - figure2_kmeans_clustering.png")
print(" - figure3_feature_importance.png")
