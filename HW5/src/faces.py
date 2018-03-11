"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Famous Faces
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
import util
from util import *
from cluster import *

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets
    
    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """
    
    n,d = X.shape
    
    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])
    
    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """
    
    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.
    
    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed
    
    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)
    
    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]
    
    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))
    
    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part c: implement (hint: use np.random.choice)
    return np.random.choice(points, k, replace=False)
    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!
    
    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part f: implement
    initial_points = []
    points_dict = {}
    k = 0
    label = set([])
    for p in points:
        label.add(p.label)
    k = len(label)
    for i in range(1, k+1):
        points_dict[i] = []
    for p in points:
        points_dict[p.label].append(p)

    for i in range(1, k+1):
        cluster = Cluster(points_dict[i])
        initial_points.append(cluster.medoid())

    return initial_points
    ### ========== TODO : END ========== ###

def kAverages(points, k, average, init='random', plot=True):
    k_clusters = ClusterSet()


    cluster_centers = None
    if(init == 'random'):
        cluster_centers = random_init(points, k)
    elif(init == 'cheat'):
        cluster_centers = cheat_init(points)


    for center in cluster_centers:
        cluster = Cluster([center]) # Create the cluster
        k_clusters.add(cluster)

    iteration_num = 0 
    while True:
        iteration_num += 1
        new_cluster_assignments = {}
        centers = average(k_clusters)

        print('%d & (%f,%f) & (%f,%f) & (%f,%f) \\\\' % (iteration_num, centers[0].attrs[0], centers[0].attrs[1], centers[1].attrs[0], centers[1].attrs[1], centers[2].attrs[0], centers[2].attrs[1]))

        for i in range(k):
            new_cluster_assignments[i] = []

        for p in points:
            min_dist_index = 0
            min_dist = None
            for i, center in enumerate(centers):
                dist_to_center = p.distance(center)
                if min_dist == None or dist_to_center < min_dist:
                    min_dist_index = i
                    min_dist = dist_to_center
            
            new_cluster_assignments[min_dist_index].append(p)
        
        new_cluster = ClusterSet()

        for i in range(k):
            cluster = Cluster(new_cluster_assignments[i])    
            new_cluster.add(cluster)

        if(new_cluster.equivalent(k_clusters)):
            break
        else:
            k_clusters = new_cluster
        if(plot):
            title_string = 'Iteration Number: %d' % iteration_num
            plot_clusters(k_clusters, title_string, average)
    if plot:
        plot_clusters(k_clusters, 'Final Clustering', average)

    return k_clusters


def kMeans(points, k, init='random', plot=False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.
    
    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable: 
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm
    
    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """
    
    ### ========== TODO : START ========== ###
    # part c: implement
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to create
    #       a new ClusterSet object and update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).
    return kAverages(points, k, ClusterSet.centroids, init=init, plot=plot)
    ### ========== TODO : END ========== ###


def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part e: implement
    return kAverages(points, k, ClusterSet.medoids, init=init, plot=plot)
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    
    ### ========== TODO : START ========== ###
    # part d, part e, part f: cluster toy dataset
    np.random.seed(1234)
    points = generate_points_2d(20) 
    clusters = kMeans(points, 3, init='random', plot=False)
    clusters = kMedoids(points, 3, init='random', plot=True)
    ### ========== TODO : END ========== ###
    
    


if __name__ == "__main__" :
    main()
