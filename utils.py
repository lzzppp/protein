import torch
import random
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min

def k_medoids(distance_matrix, k, max_iterations=100, random_state=None):
    np.random.seed(random_state)
    
    # Step 1: Initialize the K cluster centroids
    centroids_indices = np.random.choice(distance_matrix.shape[0], k, replace=False)
    
    for _ in range(max_iterations):
        # Step 2: Assign each sequence to the nearest centroid
        labels, _ = pairwise_distances_argmin_min(distance_matrix[:, centroids_indices], metric='precomputed')
        
        # Step 3: Update the centroids
        new_centroids_indices = np.zeros(k, dtype=int)
        for cluster_idx in range(k):
            cluster_member_indices = np.where(labels == cluster_idx)[0]
            cluster_distances = distance_matrix[np.ix_(cluster_member_indices, cluster_member_indices)]
            medoid_idx = cluster_member_indices[np.argmin(cluster_distances.sum(axis=1))]
            new_centroids_indices[cluster_idx] = medoid_idx
        
        # Step 4: Check for convergence
        if np.array_equal(centroids_indices, new_centroids_indices):
            break
        centroids_indices = new_centroids_indices
    
    # Step 5: Output the final clustering
    return labels, centroids_indices

def most_representative_examples(distance_matrix, labels, k):
    representatives = np.zeros(k, dtype=int)

    for cluster_idx in range(k):
        cluster_member_indices = np.where(labels == cluster_idx)[0]
        cluster_distances = distance_matrix[np.ix_(cluster_member_indices, cluster_member_indices)]
        avg_distances = cluster_distances.mean(axis=1)
        most_representative_idx = cluster_member_indices[np.argmin(avg_distances)]
        representatives[cluster_idx] = most_representative_idx

    return representatives

def get_repre_sequences(distance_matrix, k, random_state=None):
    kmedoids = KMedoids(n_clusters=k, metric='precomputed', init='k-medoids++', random_state=random_state, max_iter=10000)
    kmedoids.fit(distance_matrix)
    repre_sequence_ids = most_representative_examples(distance_matrix, kmedoids.labels_, k)
    return repre_sequence_ids

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def get_distance_matrix(distance_matrix, aids, bids):
    return distance_matrix[np.ix_(aids, bids)]

def get_distance_matrix_with_postprocess(distance_matrix, aids, bids):
    distance_matrix = distance_matrix[np.ix_(aids, bids)]
    return torch.FloatTensor(distance_matrix)

def get_distance_matrix_from_embeddings(query_embeddings, gallery_embeddings):
    return pairwise_distances(query_embeddings, gallery_embeddings, metric='euclidean')

def pairwise_distances(query_embeddings, gallery_embeddings, metric='euclidean'):
    if metric == 'cosine':
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        gallery_embeddings = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
        return 1 - np.matmul(query_embeddings, gallery_embeddings.T)
    elif metric == 'euclidean':
        return np.linalg.norm(query_embeddings[:, np.newaxis, :] - gallery_embeddings[np.newaxis, :, :], axis=2)
    else:
        raise ValueError('Invalid metric: %s' % metric)
    
def get_metrics(predict_distance_matrix, target_distance_matrix, topks=[1, 5, 10]):
    predict_labels = np.argsort(predict_distance_matrix, axis=1)
    target_labels = np.argsort(target_distance_matrix, axis=1)

    metrics = {}
    for topk in topks:
        metrics['top%d' % topk] = round(topk_accuracy(predict_labels, target_labels, topk), 2)
    return metrics

def topk_accuracy(predict_labels, target_labels, topk):
    num_correct = 0
    for predict_label, target_label in zip(predict_labels, target_labels):
        for label in predict_label[:topk]:
            if label in target_label[:topk]:
                num_correct += 1
    return num_correct / (len(predict_labels) * topk * 1.0)

def mean_average_precision(predict_labels, target_labels):
    average_precisions = []
    for predict_label, target_label in zip(predict_labels, target_labels):
        average_precision = 0
        num_correct = 0
        for i, label in enumerate(predict_label):
            if label in target_label:
                num_correct += 1
                average_precision += num_correct / (i + 1)
        average_precisions.append(average_precision / len(target_label))
    return np.mean(average_precisions)

blosum_matrix = np.array([
    [4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2] ,
    [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2] ,
    [-2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3] ,
    [-1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -2, 0, -1, 2, 0, 0, -1, -2, -3, -2] ,
    [-2, -2, -3, -3, 6, -3, -1, 0, -3, 0, 0, -3, -4, -3, -3, -2, -2, -1, 1, 3] ,
    [0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -3, 0, -2, -2, -2, 0, -2, -3, -2, -3] ,
    [-2, -3, -1, 0, -1, -2, 8, -3, -1, -3, -2, 1, -2, 0, 0, -1, -2, -3, -2, 2] ,
    [-1, -1, -3, -3, 0, -4, -3, 4, -3, -2, 1, -3, -3, -3, -3, -2, -1, 3, -3, -1] ,
    [-1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -1, 0, -1, 1, 2, 0, -1, -2, -3, -2] ,
    [-1, -1, -4, -3, 0, -4, -3, -2, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1] ,
    [-1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 5, -2, -2, 0, -1, -1, -1, 1, -1, -1] ,
    [-2, -3, 1, 0, -3, 0, 1, -3, 0, -3, -2, 6, -2, 0, 0, 1, 0, -3, -4, -2] ,
    [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2, 7, -1, -2, -1, -1, -2, -4, -3] ,
    [-1, -3, 0, 2, -3, -2, 0, -3, 1, -2, 0, 0, -1, 5, 1, 0, -1, -2, -2, -1] ,
    [-1, -3, -2, 0, -3, -2, 0, -3, 2, -2, -1, 0, -2, 1, 5, -1, -1, -3, -3, -2] ,
    [1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 1, -1, 0, -1, 4, 1, -2, -3, -2] ,
    [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 1, 5, 0, -2, -2] ,
    [0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, -2, -2, -3, -2, 0, 4, -3, -1] ,
    [-3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, 2] ,
    [-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7] ,
]) / 11