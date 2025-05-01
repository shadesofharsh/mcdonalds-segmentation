
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess(path="mcdonalds.csv"):
    # Load
    df = pd.read_csv(path)
    # Encode first 11 Yes/No columns → 0/1
    MD = df.iloc[:, 0:11].replace("Yes", 1).replace("No", 0)
    return df, MD

def do_pca(MD, n_components=2):
    scaler = StandardScaler()
    X = scaler.fit_transform(MD)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    # Summary table
    sd = np.sqrt(pca.explained_variance_)
    pv = pca.explained_variance_ratio_
    idx = [f"PC{i+1}" for i in range(len(sd))]
    summary = pd.DataFrame({
        "Standard deviation": sd,
        "Proportion of Variance": pv,
        "Cumulative Proportion": pv.cumsum()
    }, index=idx)
    return X, pca, X_pca, summary

def plot_scree(summary):
    plt.figure(figsize=(6,4))
    plt.bar(summary.index, summary["Proportion of Variance"])
    plt.plot(summary.index, summary["Cumulative Proportion"], marker='o')
    plt.ylabel("Variance")
    plt.title("Scree Plot & Cumulative Variance")
    plt.tight_layout()
    plt.savefig("scree_plot.png")
    plt.show()
    

def biplot(score, coeff, labels):
    # score: (n_samples,2), coeff:(n_features,2)
    xs, ys = score[:,0], score[:,1]
    plt.figure(figsize=(8,6))
    plt.scatter(xs, ys, alpha=0.5)
    for i,(xvec,yvec) in enumerate(coeff):
        plt.arrow(0,0, xvec, yvec, color='r', alpha=0.5)
        plt.text(xvec*1.1, yvec*1.1, labels[i], 
                 ha='center', va='center')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid(True)
    plt.title("PCA Biplot")
    plt.tight_layout()
    plt.savefig("pca_biplot.png")
    plt.show()

def elbow_plot(X, k_max=8, n_init=10):
    inertias = []
    ks = list(range(1, k_max+1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=42).fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(6,4))
    plt.plot(ks, inertias, '-o')
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Plot")
    plt.xticks(ks)
    plt.tight_layout()
    plt.savefig("elbow_plot.png")
    plt.show()

def bootstrap_stability(X, k_range=range(2,9), n_iter=50):
    stability = {}
    base_models = {k: KMeans(n_clusters=k, random_state=0).fit(X).labels_ 
                   for k in k_range}
    for k in k_range:
        scores = []
        base_labels = base_models[k]
        for _ in range(n_iter):
            Xb, idx = resample(X, replace=True, random_state=None, return_indices=True)
            labels_b = KMeans(n_clusters=k, random_state=0).fit_predict(Xb)
            ari = adjusted_rand_score(base_labels[idx], labels_b)
            scores.append(ari)
        stability[k] = scores
    # Plot
    plt.figure(figsize=(6,4))
    plt.boxplot([stability[k] for k in sorted(stability)], 
                labels=sorted(stability))
    plt.xlabel("k")
    plt.ylabel("Adjusted Rand Index")
    plt.title("Bootstrap Cluster Stability")
    plt.tight_layout()
    plt.savefig("bootstrap_stability.png")
    plt.show()

def similarity_histograms(X, clusters, centroids, k):
    sim_data = []
    for cid in range(k):
        members = X[clusters == cid]
        centroid = centroids[cid].reshape(1,-1)
        sims = cosine_similarity(members, centroid).flatten()
        sim_data.append(sims)
    # 2×2 grid
    rows = int(np.ceil(k/2))
    fig, axs = plt.subplots(rows, 2, figsize=(12, 4*rows))
    axs = axs.flatten()
    for i, sims in enumerate(sim_data):
        axs[i].hist(sims, bins=10, edgecolor='k')
        axs[i].set_title(f"Cluster {i}")
        axs[i].set_xlabel("Cosine Similarity")
        axs[i].set_ylabel("Frequency")
    # hide unused axes
    for j in range(k, len(axs)):
        axs[j].axis('off')
    plt.tight_layout()
    plt.savefig("similarity_histograms.png")
    plt.show()

if __name__ == "__main__":
    # 1. Load & preprocess
    df, MD = load_and_preprocess("mcdonalds.csv")

    # 2. PCA
    X, pca, X_pca, summary = do_pca(MD, n_components=2)
    print(summary)

    # 3. Scree plot & cumulative
    plot_scree(summary)

    # 4. Biplot
    biplot(X_pca, pca.components_.T, MD.columns.tolist())

    # 5. Elbow
    elbow_plot(X, k_max=8)

    # 6. Stability
    bootstrap_stability(X, k_range=range(2,9), n_iter=50)

    # 7. KMeans final and similarity histograms
    k_final = 4
    km = KMeans(n_clusters=k_final, random_state=42).fit(X)
    df['cluster'] = km.labels_
    similarity_histograms(X, km.labels_, km.cluster_centers_, k_final)
