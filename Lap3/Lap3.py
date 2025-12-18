import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random

print("Đang tải mô hình GloVe (100 chiều)...")
glove = api.load("glove-wiki-gigaword-100")  # 100 chiều
print("Tải xong GloVe!")
print(f"Số lượng từ trong GloVe: {len(glove.index_to_key):,}")

words = random.sample(glove.index_to_key, 300)
vectors = np.array([glove[w] for w in words])
# Giảm chiều bằng PCA
pca = PCA(n_components=2)
reduced_pca = pca.fit_transform(vectors)
# Vẽ biểu đồ 2D
plt.figure(figsize=(12, 8))
plt.scatter(reduced_pca[:, 0], reduced_pca[:, 1], alpha=0.7)
for i, word in enumerate(words):
    plt.text(reduced_pca[i, 0] + 0.02, reduced_pca[i, 1] + 0.02, word, fontsize=8)
plt.title("Giảm chiều Glove")
plt.xlabel("Thành phần chính 1")
plt.ylabel("Thành phần chính 2")
plt.show()

tsne = TSNE(n_components=3, random_state=42, perplexity=30)
reduced_tsne = tsne.fit_transform(vectors)
# Vẽ 3D
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_tsne[:,0], reduced_tsne[:,1], reduced_tsne[:,2], s=30, alpha=0.7)
for i, word in enumerate(words[:40]):  # chỉ hiển thị 40 từ để dễ nhìn
    ax.text(reduced_tsne[i,0], reduced_tsne[i,1], reduced_tsne[i,2], word, fontsize=8)
ax.set_title("Trực quan hóa 3D embeddings GloVe bằng t-SNE")
plt.show()


def cosine_similarity(vec1, vec2):
    """Tính độ tương đồng cosine giữa hai vector"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def find_most_similar(word, model, top_k=10):
    """Tìm top K từ gần nghĩa nhất với từ cho trước"""
    if word not in model:
        print(f"Từ '{word}' không có trong từ điển GloVe.")
        return
    target_vec = model[word]
    similarities = {}
    for other_word in model.key_to_index.keys():
        if other_word == word:
            continue
        sim = cosine_similarity(target_vec, model[other_word])
        similarities[other_word] = sim
    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop{top_k}từ gần nghĩa nhất với'{word}': ")
    for i, (w, score) in enumerate(sorted_words[:top_k]):
        print(f"{i + 1}. {w} ({score:.4f})")
find_most_similar("king", glove, top_k=10)
find_most_similar("computer", glove, top_k=10)
find_most_similar("beautiful", glove, top_k=10)