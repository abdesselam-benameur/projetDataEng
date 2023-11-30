from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import prince
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

def dim_red(mat, p, method):
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    if method=='ACP':
        pca = prince.PCA(
            n_components=p,
            n_iter=3,
            rescale_with_mean=True,
            rescale_with_std=True,
            copy=True,
            check_input=True,
            engine='sklearn',
            random_state=42
        )
        pca = pca.fit(mat)    
        red_mat = pca.transform(mat)
        
    elif method=='TSNE':
        tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
        red_mat = tsne.fit_transform(mat)
        
    elif method=='UMAP':
        reducer = umap.UMAP(n_components=p)
        red_mat = reducer.fit_transform(mat)
    
    else:
        raise Exception("Please select one of the three methods : APC, AFC, UMAP")
    
    return red_mat


def clust(mat, k, method):
    '''
    Perform clustering

    Input:
    -----
        mat : input list 
        k : number of cluster
    Output:
    ------
        pred : list of predicted labels
    '''
    if method=='CAH':
        clustering = AgglomerativeClustering(n_clusters=k).fit(mat)
        pred = clustering.labels_

    elif method=='Kmeans':
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(mat)
        pred = kmeans.labels_

    else:
        raise Exception("Please select one of the two methods : DBSCAN, Kmeans")
    
    return pred


# main
if __name__ == '__main__':
    # import data
    ng20 = fetch_20newsgroups(subset='test')
    corpus = ng20.data[:2000]
    labels = ng20.target[:2000]
    k = len(set(labels))

    # embeddings
    # if file does not exist, create it
    if not os.path.exists('embeddings.npy'):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = model.encode(corpus)
        np.save('embeddings.npy', embeddings)
    else:
        # load embeddings
        embeddings = np.load('embeddings.npy')
        import pandas as pd
        embeddings = pd.DataFrame(embeddings)

    # create imgs folder if it does not exist
    if not os.path.exists('imgs'):
        os.makedirs('imgs')

    # Perform dimensionality reduction and clustering for each method
    dim_red_methods = ['ACP', 'TSNE', 'UMAP', None]
    clustering_algorithms = ['Kmeans', 'CAH']
    for method in dim_red_methods:
        for clust_alg in clustering_algorithms:
            if method is None:
                print(f'Method: Without dimensionality reduction, clustering: {clust_alg}') 
                red_emb = embeddings
            else:
                print(f'Method: {method}, clustering: {clust_alg}')
                # Perform dimensionality reduction
                red_emb = dim_red(embeddings, 20, method)

            # Perform clustering
            pred = clust(red_emb, k, clust_alg)

            # Evaluate clustering results
            nmi_score = normalized_mutual_info_score(pred, labels)
            ari_score = adjusted_rand_score(pred, labels)

            # Print results
            red_emb = pd.DataFrame(red_emb)
            print(f'NMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')
            # Plot results (save as png)
            plt.scatter(red_emb[0], red_emb[1], c=pred)
            plt.savefig(f'imgs/clustering_{method}_{clust_alg}.png')
            plt.close()
    
