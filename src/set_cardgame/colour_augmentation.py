import numpy as np
import cv2
from sklearn.cluster import KMeans


def reduce_with_clusters(img, n_clusters=10):
    # number of clusters
    kmeans = KMeans(n_clusters=n_clusters)

    # flatten image
    img_2d = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    
    # fit image
    kmeans = kmeans.fit(img_2d)

    # centroid values
    centroids = kmeans.cluster_centers_

    # move pixel values to nearest cluster
    distances = []
    for centroid in centroids:
        distances.append(np.linalg.norm(img_2d - centroid, axis=1))
    distances = np.array(distances).T
    closed_centroid = np.argmin(distances, axis=1)
    img_reduced = centroids[closed_centroid]
    return img_reduced.reshape(96, 128, 3), centroids


def correct_color(img_2d, n_clusters=10):
    img = img_2d.reshape(96, 128, 3)
    
    img_reduced, centroids = reduce_with_clusters(img, n_clusters=n_clusters)
    largest_centroid = centroids[np.argmax(np.linalg.norm(centroids, axis=1))]
    dist_to_white = 255-largest_centroid
    
    # normalize image
    img_norm = cv2.normalize(img_reduced, None, 0, 255, cv2.NORM_MINMAX)
    img_norm = cv2.convertScaleAbs(img_norm)
    
    # reshape to 2d
    img_norm_2d = img_norm.reshape(img.shape[0]*img.shape[1], img.shape[2])
    
    # correct to white
    img_norm_2d_new = img_norm_2d + dist_to_white
    
    # scale back to uint8
    img_norm_2d_new = cv2.convertScaleAbs(img_norm_2d_new)

    # reshape to image
    img_norm_2d_new = img_norm_2d_new.reshape(img.shape)
    
    return img_norm_2d_new


def batch_correct_color(X, n_clusters=10):
    X_2d = X.reshape(X.shape[0], -1)
    return np.apply_along_axis(correct_color, 1, X_2d, n_clusters=n_clusters)



def correct_color1(img_flat, n_clusters=10):
    
    img = img_flat.reshape(96, 128, 3)
    img_2d = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    
    img_reduced, centroids = reduce_with_clusters(img, n_clusters=n_clusters)
    largest_centroid = centroids[np.argmax(np.linalg.norm(centroids, axis=1))]
    dist_to_white = 255-largest_centroid
    
    # normalize image
    img_norm = cv2.normalize(img_reduced, None, 0, 255, cv2.NORM_MINMAX)
    img_norm = cv2.convertScaleAbs(img_norm)
    
    # reshape to 2d
    img_norm_2d = img_norm.reshape(img.shape[0]*img.shape[1], img.shape[2])
    
    # correct to white
    img_2d_new = img_2d + dist_to_white
    
    # scale back to uint8
    img_2d_new = cv2.convertScaleAbs(img_2d_new)
    
    # reshape to image
    img_new = img_2d_new.reshape(img.shape)
    
    return img_new



def batch_correct_color1(X, n_clusters=10):
    X_2d = X.reshape(X.shape[0], -1)
    return np.apply_along_axis(correct_color1, 1, X_2d, n_clusters=n_clusters)
