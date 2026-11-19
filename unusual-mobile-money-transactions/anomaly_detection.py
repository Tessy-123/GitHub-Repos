import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib


np.random.seed(42)


def generate_synthetic_data(n_samples=5000, anomaly_fraction=0.05):
    """Create a synthetic mobile-money dataset with a small proportion of anomalies."""
    n_anomalies = int(n_samples * anomaly_fraction)
    n_normal = n_samples - n_anomalies

    # Normal transactions
    normal_amounts = np.random.exponential(scale=150, size=n_normal) + 10
    normal_old_bal = np.random.uniform(500, 10000, size=n_normal)
    normal_new_bal = np.maximum(0, normal_old_bal - normal_amounts)
    normal_types = np.random.choice(
        ["TRANSFER", "CASH_OUT", "PAYMENT", "DEPOSIT"],
        size=n_normal,
        p=[0.3, 0.3, 0.2, 0.2],
    )

    # Anomalous transactions
    anomaly_amounts = np.random.uniform(15000, 50000, size=n_anomalies)
    anomaly_old_bal = np.random.uniform(100, 2000, size=n_anomalies)
    anomaly_new_bal = anomaly_old_bal + np.random.uniform(5000, 20000, size=n_anomalies)
    anomaly_types = np.random.choice(["TRANSFER", "CASH_OUT"], size=n_anomalies, p=[0.7, 0.3])

    amounts = np.concatenate([normal_amounts, anomaly_amounts])
    old_bal = np.concatenate([normal_old_bal, anomaly_old_bal])
    new_bal = np.concatenate([normal_new_bal, anomaly_new_bal])
    types = np.concatenate([normal_types, anomaly_types])
    labels = np.concatenate([
        np.zeros(n_normal, dtype=int),
        np.ones(n_anomalies, dtype=int),
    ])

    df = pd.DataFrame(
        {
            "type": types,
            "amount": amounts,
            "oldbalanceOrg": old_bal,
            "newbalanceOrig": new_bal,
            "balance_change": new_bal - old_bal,
            "actual_anomaly": labels,
        }
    )

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def main():
    print("--- Step 1: Generating Synthetic Mobile Money Dataset ---")
    df = generate_synthetic_data(n_samples=5000, anomaly_fraction=0.05)
    print(f"Dataset created: {df.shape[0]} records.")
    print(f"Actual anomalous transactions: {df['actual_anomaly'].sum()}")

    print("\n--- Step 2: Data Preprocessing ---")
    df_encoded = pd.get_dummies(df, columns=["type"], drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    print("\n--- Step 3: Model 1 - Isolation Forest ---")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_predictions = iso_forest.fit_predict(X_scaled)
    df["iso_forest_anomaly"] = (iso_predictions == -1).astype(int)
    print(f"Isolation Forest flagged anomalies: {df['iso_forest_anomaly'].sum()}")

    print("\n--- Step 4: Model 2 - K-Means Clustering ---")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(X_scaled - centroids[df["kmeans_cluster"]], axis=1)
    threshold = np.percentile(distances, 95)
    df["kmeans_anomaly"] = (distances > threshold).astype(int)
    print(f"K-Means flagged anomalies: {df['kmeans_anomaly'].sum()}")
    print(f"K-Means silhouette score: {silhouette_score(X_scaled, df['kmeans_cluster']):.4f}")

    print("\n--- Step 5: Visualizing Results ---")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["pca_1"] = X_pca[:, 0]
    df["pca_2"] = X_pca[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(
        data=df,
        x="pca_1",
        y="pca_2",
        hue="iso_forest_anomaly",
        palette={0: "blue", 1: "red"},
        alpha=0.6,
        ax=axes[0],
    )
    axes[0].set_title("Isolation Forest Anomaly Detection")
    axes[0].set_xlabel("PCA Component 1")
    axes[0].set_ylabel("PCA Component 2")

    sns.scatterplot(
        data=df,
        x="pca_1",
        y="pca_2",
        hue="kmeans_anomaly",
        palette={0: "green", 1: "orange"},
        alpha=0.6,
        ax=axes[1],
    )
    axes[1].set_title("K-Means Distance-Based Anomaly Detection")
    axes[1].set_xlabel("PCA Component 1")
    axes[1].set_ylabel("PCA Component 2")

    plt.tight_layout()
    plt.savefig("anomaly_detection_results.png", dpi=300)
    print("Saved visualization chart as 'anomaly_detection_results.png'")

    print("\n--- Step 6: Saving Models & Artifacts ---")
    joblib.dump(iso_forest, "iso_forest_model.pkl")
    joblib.dump(kmeans, "kmeans_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(pca, "pca_model.pkl")
    df.to_csv("mobile_money_anomalies_output.csv", index=False)

    print("Model training complete. All output files saved successfully!")


if __name__ == "__main__":
    main()
