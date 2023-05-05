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
    
    repre_sequences = pickle.load(open("./cache/repre_sequences", 'rb'))
    
    repre_sequence_ids = [id for id, seq in repre_sequences]
    
    query_sequences, gallery_sequences = test_sequences[:len(query_embeddings)], test_sequences[len(query_embeddings):]
    for i in range(len(query_embeddings)):
        # print("query embedding: ", query_embeddings[i])
        target_top1 = target_distance_matrix[i].argsort()[0]
        predict_top1 = predict_distance_matrix[i].argsort()[0]

        real_query_embedding = [distance_matrix[i][rid] for rid in repre_sequence_ids]
        real_true_embedding = [distance_matrix[target_top1][rid] for rid in repre_sequence_ids]
        real_false_embedding = [distance_matrix[predict_top1][rid] for rid in repre_sequence_ids]

        if target_top1 != predict_top1:
            query_proteins.append(query_sequences[i])
            true_gallery_proteins.append(gallery_sequences[target_top1])
            false_gallery_proteins.append(gallery_sequences[predict_top1])
            query_protein_embeddings.append(query_embeddings[i])
            true_gallery_protein_embeddings.append(gallery_embeddings[target_top1])
            false_gallery_protein_embeddings.append(gallery_embeddings[predict_top1])
            query_true_target_distance.append(target_distance_matrix[i][target_top1])
            query_false_target_distance.append(target_distance_matrix[i][predict_top1])
            
            real_query_embeddings.append(real_query_embedding)
            real_true_embeddings.append(real_true_embedding)
            real_false_embeddings.append(real_false_embedding)
    
    # random_choice = random.sample(range(len(query_proteins)), 5)
    # print("random choice: ", random_choice)
    # asas
    random_choice = [325, 62, 261, 184, 321]
    
    for qi in random_choice:
        print("\n\n\n")
        print("query protein: ", query_proteins[qi])
        print("\n")
        print("true gallery protein: ", true_gallery_proteins[qi])
        print("\n")
        print("false gallery protein: ", false_gallery_proteins[qi])
        print("\n")
        print("query protein embedding: \t", query_protein_embeddings[qi])
        print("true gallery protein embedding: \t", true_gallery_protein_embeddings[qi])
        print("false gallery protein embedding: \t", false_gallery_protein_embeddings[qi])
        print("\n")
        print("\n")
        print("real query protein embedding: \t", real_query_embeddings[qi])
        print("real true gallery protein embedding: \t", real_true_embeddings[qi])
        print("real false gallery protein embedding: \t", real_false_embeddings[qi])
        print("\n")
        print("query with true target distance: \t", query_true_target_distance[qi])
        print("query with false target distance: \t", query_false_target_distance[qi])
        print("\n\n\n")
    
    # analysis embeddings of query and gallery proteins
    maxs = np.max(query_embeddings, axis=0)
    mins = np.min(query_embeddings, axis=0)
    means = np.mean(query_embeddings, axis=0)
    stds = np.std(query_embeddings, axis=0)
    print("query protein embeddings max: \t", maxs)
    print("query protein embeddings min: \t", mins)
    print("query protein embeddings mean: \t", means)
    print("query protein embeddings std: \t", stds)
    maxs = np.max(gallery_embeddings, axis=0)
    mins = np.min(gallery_embeddings, axis=0)
    means = np.mean(gallery_embeddings, axis=0)
    stds = np.std(gallery_embeddings, axis=0)
    print("\ngallery protein embeddings max: \t", maxs)
    print("gallery protein embeddings min: \t", mins)
    print("gallery protein embeddings mean: \t", means)
    print("gallery protein embeddings std: \t", stds)
    
    embeddings = np.concatenate((query_embeddings, gallery_embeddings), axis=0)
    target_embeddings = []
    for i in range(len(embeddings)):
        target_embedding = [distance_matrix[i][rid] for rid in repre_sequence_ids]
        target_embeddings.append(target_embedding)
    target_embeddings = np.array(target_embeddings)
    mae_loss = np.mean(np.abs(embeddings - target_embeddings), axis=1)
    print("mae loss: ", mae_loss.shape)
    
    topks = [1, 5, 10, 50, 100, 500]
    query_target_embeddings = target_embeddings[:len(query_embeddings)]
    gallery_target_embeddings = target_embeddings[len(query_embeddings):]
    
    predict_distance_matrix = get_distance_matrix_from_embeddings(query_target_embeddings, gallery_target_embeddings)
    print("predict distance matrix: ", predict_distance_matrix.shape)
    print("target distance matrix: ", target_distance_matrix.shape)
    
    # calculate metrics
    metrics = get_metrics(predict_distance_matrix, distance_matrix[:500,500:3000], topks=topks)
    print("metrics: ", metrics, "\n")
    
    # repre_embeddings = pickle.load(open("./cache/repre_embeddings", 'rb'))
    
    repre_embeddings = [embeddings[rid] for rid in repre_sequence_ids]
    repre_embeddings = np.array(repre_embeddings)
    pickle.dump(repre_embeddings, open("./cache/repre_embeddings", 'wb'))
    
    # kmeans = KMeans(n_clusters=2)
    # kmeans.fit(embeddings)
    
   # Get the cluster assignments for each data point
    # labels = kmeans.labels_

    # Get the cluster centroids
    # centroids = kmeans.cluster_centers_

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # print("min: ", np.min(mse_loss))
    # print("max: ", np.max(mse_loss))
    # thermal_values_normalized = (mse_loss - np.min(mse_loss)) / (np.max(mse_loss) - np.min(mse_loss))
    # print("thermal_values_normalized: ", thermal_values_normalized.shape)
    # Plot the data points, colored by their cluster assignment
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=mae_loss, cmap='hot', label="data points", vmin=np.min(mae_loss), vmax=np.max(mae_loss))

    # Plot the centroids
    # ax.scatter(centroids[:, 0], centroids[:, 1],
    #         c='red', marker='x', s=100, label="centroids")
    
    # Plot the representative sequences
    ax.scatter(repre_embeddings[:, 0], repre_embeddings[:, 1], repre_embeddings[:, 2],
            c='green', marker='s', s=100, label="anchors")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Thermal Values')

    # ax.set_title('K-means Clustering')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.legend()

    # plt.show()
    plt.savefig("./cache/kmeans_cross_train_3.png")
    
    plt.cla()
    plt.clf()
    
    # t-SNE
    # umap_model = umap.UMAP(n_components=2, metric='precomputed', random_state=42)
    # umap_coordinates = umap_model.fit_transform(distance_matrix)

    # plt.figure()
    # plt.scatter(umap_coordinates[:, 0], umap_coordinates[:, 1])
    # plt.title('UMAP Visualization')
    # plt.savefig("./cache/tsne.png")