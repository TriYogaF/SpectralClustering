import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import silhouette_score

# Download stopwords jika belum diunduh
nltk.download('stopwords')

url = 'https://raw.githubusercontent.com/TriYogaF/dataset/main/elden_ring_steam_reviews.csv'
df = pd.read_csv(url)

# Hapus baris yang memiliki nilai null atau kosong jika ada
df.dropna(subset=['review'], inplace=True)

# Preprocessing tambahan: membersihkan teks dan menghapus kata-kata umum
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['review'] = df['review'].apply(preprocess_text)

# Ekstraksi fitur teks menggunakan TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['review'])

# Reduksi dimensi dengan menggunakan PCA
n_components = 2
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# Hitung similarity Gaussian dari data PCA
sigma = 1.0
gaussian_similarity = np.exp(-pairwise_distances(pca_result, squared=True) / (2 * (sigma ** 2)))

# Terapkan Spectral Clustering pada similarity matrix
n_clusters = 2  # Ubah menjadi 2-8 kluster
spectral_cluster = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
cluster_labels = spectral_cluster.fit_predict(gaussian_similarity)

# Gabungkan label cluster dengan data awal
df['cluster'] = cluster_labels

# Visualisasi scatter plot
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orchid']

for cluster_id in range(n_clusters):
    cluster_data = pca_result[df['cluster'] == cluster_id]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[cluster_id], label=f'Cluster {cluster_id}')

plt.legend()
plt.title('Scatter Plot Hasil Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Analisis setiap cluster
clustered_reviews = {}
for cluster_id in range(n_clusters):
    cluster_data = df[df['cluster'] == cluster_id]
    clustered_reviews[f'Cluster {cluster_id}'] = cluster_data['review'].tolist()

for cluster_id, reviews in clustered_reviews.items():
    print(f'Analisis Cluster {cluster_id}:')
    print('Jumlah Ulasan:', len(reviews))
    display('Contoh Ulasan:', reviews[0:5])  # Cetak contoh ulasan 1-6 dalam cluster
    print('\n')

# Menghitung Silhouette Score
silhouette_avg = silhouette_score(gaussian_similarity, cluster_labels)
print(f'Silhouette Score: {silhouette_avg}')




