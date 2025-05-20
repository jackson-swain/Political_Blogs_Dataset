import numpy as np
from os.path import abspath, exists
import matplotlib.pyplot as plt

#Using the same logic as KMeans, copy over the useful methods rather than calling the class to avoid unecessary steps
class KMeansImpl:
    def __init__(self, max_iterations=100, tolerance=1e-4, random=10):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random = random
        np.random.seed(self.random)
    
    def centroid_initialization(self, pixels, num_clusters):
        ##Randomly select centroid initial locations based on the random state seed

        num_pixels = pixels.shape[0]
        random_pixel_s = np.random.choice(num_pixels, num_clusters)
        return pixels[random_pixel_s].copy()
    
    def assign_clusters(self, pixels, centroids, normalized_choice):
        ##Assign each pixel in the image to the closest centroid
        ##Manhattan distance is normalized_choice=1 and Euclidean distance is normalized_choice=2
        ##Adjusting for vectorization implementation to speed up algorithm
        
        ##To use vectorization, reshape the pixels/matrices
        expanded_pixels = pixels[:, np.newaxis, :]
        expanded_centroids = centroids[np.newaxis, :, :]

        if(normalized_choice == 1):
            distance = np.sum(np.abs(expanded_pixels - expanded_centroids), axis=2)
        else:
            distance = np.sum((expanded_pixels - expanded_centroids)**2, axis=2)

        cluster_assignment = np.argmin(distance, axis = 1)
        return cluster_assignment
    
    def move_centroids(self, pixels, cluster_assignment, num_clusters):
        ##Using the mean of the clusters, update that cluster centroid
        ##Adjusting for vectorization implementation to speed up algorithm

        ##Initialize the array, divide the cluster assignments into seperate matrices, and count the pixels in each cluster
        centroids = np.zeros((num_clusters, pixels.shape[1]))
        cluster_matrix = np.zeros((pixels.shape[0], num_clusters))
        cluster_matrix[np.arange(pixels.shape[0]), cluster_assignment]=1
        cluster_size = np.sum(cluster_matrix, axis=0)

        ##Find the new centroids
        for j in range(pixels.shape[1]):
            sum_pixels = np.sum(cluster_matrix * pixels[:,j][:,np.newaxis], axis=0)
            centroids[:,j] = np.divide(sum_pixels, cluster_size, out=np.zeros_like(sum_pixels), where=cluster_size!=0)
        
        empty_clusters = np.where(cluster_size == 0)[0].tolist()
        return centroids, empty_clusters

    def empty_clusters(self, pixels, centroids, cluster_assignment, empty_cluster):
        ##Assign empty clusters to the largest exsisting centroid
        ##Adjusting for vectorization implementation to speed up algorithm

        ##Copy in the new centroids to use the latest position
        new_centroids = centroids.copy()

        if empty_cluster:
            ##Find the size of each cluster
            cluster_sizes = np.bincount(cluster_assignment, minlength=len(centroids))
            for empty_pixels in empty_cluster:
                ##Define the largest cluster
                largest_cluster = np.argmax(cluster_sizes)
                ##Define the indicies of the largest cluster and 
                largest_cluster_index = np.where(cluster_assignment == largest_cluster)[0]
                if len(largest_cluster_index) > 0:
                    random_index = np.random.choice(largest_cluster_index)
                    new_centroids[empty_pixels] = pixels[random_index]

        return new_centroids
    
    def cluster(self, data, num_clusters, norm_distance=2):
        ##Modified the compress() function

        result = {
            "class": None,
            "centroid": None,
            "number_of_iterations": None
        }

        ##Run centroid_initialization to start
        centroids = self.centroid_initialization(data, num_clusters)

        ##Iteration #0 and define prev_centroids to determine convergence
        iterations = 0
        prev_centroids = None

        while (iterations < self.max_iterations):
            ##Assign pixels to clusters and update the centroids
            cluster_assignments = self.assign_clusters(data, centroids, norm_distance)
            new_centroids, empty_clusters = self.move_centroids(data, cluster_assignments, num_clusters)

            ##Deal with empty clusters
            if empty_clusters:
                new_centroids = self.empty_clusters(data, new_centroids, cluster_assignments, empty_clusters)

            ##Check for convergence based on the distance the centroids moved
            if prev_centroids is not None:
                ##Manhattan distance
                if norm_distance == 1:
                    delta = np.sum(np.abs(new_centroids - prev_centroids))
                ##Euclidean distance
                else:
                    delta = np.sum((new_centroids - prev_centroids)**2)

                ##Break if the delta does not exceed the tolerance
                if delta < self.tolerance:
                    break

            ##Update for the next round
            prev_centroids = new_centroids.copy()
            centroids = new_centroids.copy()
            iterations = iterations + 1

        result["class"] = cluster_assignments.reshape(-1,1)
        result["centroid"] = centroids
        result["number_of_iterations"] = iterations+1

        return result

def load_process_data(nodes='nodes.txt', edges="edges.txt"):
    ##Read and process nodes.txt and edges.txt

    labels_id = {}
    nodes_id = []
    ##Read each line in nodes.txt and store the ID as well as the label (0 or 1) into an array
    with open(nodes, 'r') as f:
        for line in f:
            sections = line.strip().split()
            if (len(sections) >= 3):
                node_id = int(sections[0])
                label = int(sections[2])
                labels_id[node_id] = label
                nodes_id.append(node_id)
        
    ##Map the node IDs to indices and vice versa
    node_to_index = {node_id: index for index, node_id in enumerate(nodes_id)}
    index_to_node = {index: node_id for node_id, index in node_to_index.items()}

    ##Create a nxn Adjency matrix
    n = len(node_to_index)
    A = np.zeros((n,n), dtype=int)

    ##Filter through edges.txt and update the Adjency matrix
    with open(edges, 'r') as f:
        for line in f:
            sections = line.strip().split()
            if (len(sections) >= 2):
                end_one = int(sections[0])
                end_two = int(sections[1])
                if(end_one in node_to_index) and (end_two in node_to_index):
                    i = node_to_index[end_one]
                    j = node_to_index[end_two]
                    ##A[i][j] = A[j][i]
                    A[i][j] = 1
                    A[j][i] = 1
        
    ##Remove isolated nodes
    ##Define all non_isolated nodes
    degrees = np.sum(A, axis=1)
    non_isolated = np.where(degrees>0)[0]

    ##Keep all non_isolated nodes in A
    A = A[non_isolated][:,non_isolated]
    clean_node_ids = [index_to_node[i] for i in non_isolated]
    labels = np.array([labels_id[node_id] for node_id in clean_node_ids])

    return A, labels

def laplacian(A):
    ##Create a symmetric normalized matrix L

    ##L=D^-0.5 * a * D^-0.5
    D = np.diag(1/np.sqrt(np.sum(A, axis=1)))
    ##Matrix multiplication
    result = D@A@D
    return result
    
def spectral_clustering(A, k):
    ##Using the normalized Laplacian, find the eigenvectors
        
    ##Define the Laplacian of the Adjancy matrix and define the eigenvectors/values
    L = laplacian(A) 
    eigenvalues,eigenvectors = np.linalg.eig(L)

    ##Only use the biggest eigenvalues
    index = np.argsort(eigenvalues)[-k:]
    row_vector = eigenvectors[:,index].real

    ##Normalize each row vector
    row_vector = row_vector / np.linalg.norm(row_vector, axis=1, keepdims=True)

    ##Run KMeans on the smaller dataset to find clusters
    kmeans = KMeansImpl(max_iterations=100, tolerance=1e-4, random=10)
    final = kmeans.cluster(row_vector, k, norm_distance=2)

    return final["class"].flatten()
    
def compute_mismatch_rates(pred_cluster, one_labels, k):
    ##Compare the predicted cluster assignments with the labeled dataset
    
    mismatch_rates = []
    majority_labels = []

    ##Cycle thru all nodes assigned to each cluster
    for cluster_id in range(k):
        indices = np.where(pred_cluster == cluster_id)[0]

        ##Find the majority label in each cluster 
        cluster_labels = one_labels[indices]
        unique_labels, count_labels = np.unique(cluster_labels, return_counts=True)
        majority_label = unique_labels[np.argmax(count_labels)]
        majority_labels.append(majority_label)
        
        ##Find the number of mismatches
        mismatches = np.sum(cluster_labels != majority_label)
        mismatch_rate = mismatches/len(cluster_labels)
        mismatch_rates.append(mismatch_rate)

    return majority_labels, mismatch_rates

def run():
    ##Run the code for pt 1
    A, labels = load_process_data()
    for k in [2,5,10,30,50]:
        print(f"\nClustering with k = {k}")
        pred_clusters = spectral_clustering(A,k)
        majority_labels, mismatch_rates = compute_mismatch_rates(pred_clusters, labels, k)
        for index in range(k):
            print(f"    Cluster {index},    Majority label = {majority_labels[index]},    Mismatch rate = {mismatch_rates[index]}")
        avg_rate = np.mean(mismatch_rates)
        print(f"    Average mismatch rate: {avg_rate}")

    ##Run the code for pt 2
    mismatch = []
    for k in range(2,50):
        pred_clusters = spectral_clustering(A,k)
        majority_labels, mismatch_rates = compute_mismatch_rates(pred_clusters, labels, k)
        avg_rate = np.mean(mismatch_rates)
        mismatch.append(avg_rate) 
    x = np.arange(2,50)
    
    ##Plot the results to see what the optimat k value is
    plt.figure(figsize=(10,5))
    plt.plot(x,mismatch)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Average Mismatch of Each Cluster")
    plt.grid(True)
    plt.show()

run()