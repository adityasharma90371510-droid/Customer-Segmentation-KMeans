import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("Mall_Customers.csv")

print("First 5 rows:\n")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------
# Gender Distribution
# -----------------------------

print("\nGender Distribution:")
print(df["Gender"].value_counts())

sns.countplot(data=df, x="Gender")
plt.title("Gender Distribution of Customers")
plt.show()


# -----------------------------
# Income vs Spending Behavior
# -----------------------------

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)"
)

plt.title("Customer Income vs Spending Behavior")
plt.show()


# -----------------------------
# Feature Selection
# -----------------------------

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]


# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------
# Elbow Method (Find Best K)
# -----------------------------

inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), inertia, marker="o")
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()


# -----------------------------
# KMeans Clustering
# -----------------------------

kmeans = KMeans(n_clusters=5, random_state=42)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nClustered Data Sample:\n")
print(df.head())


# -----------------------------
# Cluster Visualization
# -----------------------------

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1"
)

plt.title("Customer Segmentation using K-Means")
plt.show()


# -----------------------------
# Cluster Summary
# -----------------------------

print("\nCluster Summary:")

cluster_summary = df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()

print(cluster_summary)


# -----------------------------
# Business Insight Summary
# -----------------------------

print("\n--- Business Insight Summary ---")

print("""

Customer segmentation analysis identifies five distinct customer groups
based on annual income and spending behavior.

Key insights:

Cluster 0 → Low income, low spending customers
Cluster 1 → High income, high spending customers (Premium segment)
Cluster 2 → High income but low spending (Upsell opportunity)
Cluster 3 → Low income but high spending (Potential loyal customers)
Cluster 4 → Moderate income and spending customers

Strategic Recommendations:

• Target premium clusters with loyalty programs and exclusive offers.
• Encourage high income but low spending customers with personalized promotions.
• Retain high engagement customers through rewards and targeted marketing.

Customer segmentation helps businesses optimize marketing strategies,
improve retention, and maximize revenue.

""")