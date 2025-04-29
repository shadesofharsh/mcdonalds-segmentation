import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

# Load dataset
data = pd.read_csv("mcdonalds.csv")
print("✅ Data loaded.")
print(data.head())

# -------------------------------
# Step 1: Data Preprocessing
# -------------------------------
print("\n📦 Data Info:")
print(data.info())

# Convert Yes/No to 1/0
data = data.replace({'Yes': 1, 'No': 0})

# Label encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Save column names before scaling for analysis
original_columns = data.columns.tolist()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# -------------------------------
# Step 2: Elbow Method
# -------------------------------
inertia = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_data)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig("charts/elbow_method.png")
plt.close()
print("📉 Elbow method chart saved.")

# -------------------------------
# Step 3: Apply K-Means Clustering
# -------------------------------
optimal_k = 4  # Choose based on elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)
print("✅ K-Means clustering applied.")

# -------------------------------
# Step 4: Cluster Visualization using PCA
# -------------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
data['PCA1'] = pca_data[:, 0]
data['PCA2'] = pca_data[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=80)
plt.title('Customer Segments (PCA Clusters)')
plt.savefig("charts/pca_clusters.png")
plt.close()
print("🎨 PCA cluster chart saved.")

# -------------------------------
# Step 5: Cluster Insights
# -------------------------------
cluster_summary = data.groupby('Cluster')[original_columns].mean()
cluster_summary.to_csv("charts/cluster_summary.csv")
print("📊 Cluster summary saved to CSV.")
print(cluster_summary)

# -------------------------------
# Step 6: Save Processed Dataset
# -------------------------------
data.to_csv("charts/clustered_mcdonalds_data.csv", index=False)
