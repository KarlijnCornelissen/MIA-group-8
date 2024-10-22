import numpy as np
import scipy.io



def loading_data(k):
    #loading training and testing data
    fn = '../data/nuclei_data.mat'
    mat = scipy.io.loadmat(fn)
    test_images = mat["test_images"] # shape (24, 24, 3, 20730)
    test_y = mat["test_y"] # shape (20730, 1)
    training_images = mat["training_images"] # shape (24, 24, 3, 21910)
    training_y = mat["training_y"] # shape (21910, 1)


    #k means clustering:
def distance(x1,x2):
    """
    Input: two datapoints (images)
    Output": Euclidean distance between these two datapoints 

    """
    euclidean_distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return euclidean_distance

def k_NN(X_train, y_train, X_test, k=3): 
    all_distances=[]
    for i in in range(len(X_train)):
        distance=distance(X_train[i],X_test)
        all_distances.append((distance,y_train[i)]) #append a tuple
    all_distances.sort(key=lambda x: x[0])
    neighbours=all_distances[:k]    
    neighbour_classes=[]
    for i in neighbours:
        neighbour_class=neighbours[i][1]
        neighbour_classes.append(neighbour_class)
    predicted_class = max(set(neighbour_classes), key=neighbour_classes.count)  
    
def classify_by_k_NN(X_train,y_train,X_test,k=3):
    predicted_classes={}
    for i in range(len(X_test)):
        predicted_class = k_NN(X_train, y_train, X_test[i], k=k)
        predicted_classes[X_test[i]]=predicted_class

     return predicted_classes

def calculate_error():
    a=3

def pca_transform(X):
    n_samples = X.shape[0]
    
    X_mean = np.mean(X, axis = 0)

    X_hat = X - X_mean # Center data

    sigma_hat = 1/(n_samples-1)*X_hat.T.dot(X_hat) # Calculate covariance matrix, alternative is np.cov(X)

    U, s, V = np.linalg.svd(sigma_hat) # Do singular value decomposition to get eigen vector/values

    X_pca = U.dot(X_hat.T) # Transform dataset using eigenvectors
    
    return X_pca.T, U, s


def choosing_k(low,high):
    k_and_errors=[]
    for k in range(low,high):
        #run k NN and calculate error
        k_and_errors.append((k,error))

    k_and_errors.sort(key=lambda x:x[1])
    best_k=k_and_errors[0][0]

def comparison():
    calculate_error without PCA
    calculate_error with PCA
    if without PCA < with PCA:
        print('PCA does improve k-means clustering')