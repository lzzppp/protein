import umap
import pickle
import pprint
import random
import numpy as np
from utils import get_metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS, TSNE
from utils import get_repre_sequences, set_seeds, get_distance_matrix_with_postprocess, get_distance_matrix_from_embeddings, get_metrics

if __name__ == "__main__":
    distance_matrix = 1.0 - pickle.load(open("/mnt/sda/czh/Neuprotein/5000_needle_512/similarity_matrix_result", 'rb')) / 100.0
    train_distance_matrixs = distance_matrix[:3000, :3000]
    seq_list = pickle.load(open("/mnt/sda/czh/Neuprotein/5000_needle_512/seq_list", 'rb'))
    train_sequences = seq_list[:3000]
    
    target_distances = pickle.load(open("./cache/target_distances", 'rb'))
    predict_distances = pickle.load(open("./cache/predict_distances", 'rb'))
    target_distance_matrix = pickle.load(open("./cache/target_distance_matrix", 'rb'))
    predict_distance_matrix = pickle.load(open("./cache/predict_distance_matrix", 'rb'))
    test_sequences = pickle.load(open("./cache/test_sequences", 'rb'))
    query_embeddings = pickle.load(open("./cache/query_protein_embeddings", 'rb'))
    gallery_embeddings = pickle.load(open("./cache/gallery_protein_embeddings", 'rb'))
    
    metrics = get_metrics(predict_distance_matrix, target_distance_matrix, topks=[1, 5, 10, 20, 50, 100])
    
    query_proteins, true_gallery_proteins, false_gallery_proteins = [], [], []
    query_protein_embeddings, true_gallery_protein_embeddings, false_gallery_protein_embeddings = [], [], []
    query_true_target_distance, query_false_target_distance = [], []
    real_query_embeddings, real_true_embeddings, real_false_embeddings = [], [], []
    
    repre_sequence_ids = get_repre_sequences(train_distance_matrixs, 2000, random_state=0)
    repre_sequences = [[repre_id, train_sequences[repre_id]] for repre_id in repre_sequence_ids]
    repre_sequences = pickle.load(open("./cache/repre_sequences", 'rb'))
    
    embeddings = np.concatenate((query_embeddings, gallery_embeddings), axis=0)
    target_embeddings = []
    for i in range(len(embeddings)):
        target_embedding = [distance_matrix[i][rid] for rid in repre_sequence_ids]
        target_embeddings.append(target_embedding)
    target_embeddings = np.array(target_embeddings)
    print("target embeddings: ", target_embeddings.shape)
    
    topks = [1, 5, 10, 50, 100, 500]
    query_target_embeddings = target_embeddings[:len(query_embeddings)]
    gallery_target_embeddings = target_embeddings[len(query_embeddings):]
    
    predict_distance_matrix = get_distance_matrix_from_embeddings(query_target_embeddings, gallery_target_embeddings)
    print("predict distance matrix: ", predict_distance_matrix.shape)
    print("target distance matrix: ", target_distance_matrix.shape)
    
    # calculate metrics
    metrics = get_metrics(predict_distance_matrix, distance_matrix[:500,500:3000], topks=topks)
    print("metrics: ", metrics, "\n")