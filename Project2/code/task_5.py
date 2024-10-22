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
    """
    Input: 
    Output:  A list of 
    """
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
    return predicted_class

def classify_by_k_NN(X_train,y_train,X_test,k=3):
    predicted_classes={}
    for i in range(len(X_test)):
        predicted_class = k_NN(X_train, y_train, X_test[i], k=k)
        predicted_classes[X_test[i]]=predicted_class

     return predicted_classes

def calculate_error(predicted_classes, y_test):
    """
    Input:
    - predicted_classes: dictionary where keys are test images and values are predicted class labels
    - y_test: true class labels for the test set

    Output: 
    - error rate
    - accuracy
    """

    correct = 0
    total = len(y_test)

    for i, true_class in enumerate(y_test):
        # Assuming X_test[i] is in the same order as y_test
        predicted_class = predicted_classes[i]  # Access the predicted class for the i-th test image
        if predicted_class == true_class:
            correct += 1

    accuracy = correct / total
   # error_rate = 1 - accuracy

    return accuracy


def pca(X):
    n_samples = X.shape[0]
    
    X_mean = np.mean(X, axis = 0)

    X_hat = X - X_mean # Center data

    sigma_hat = 1/(n_samples-1)*X_hat.T.dot(X_hat) # Calculate covariance matrix, alternative is np.cov(X)

    U, s, V = np.linalg.svd(sigma_hat) # Do singular value decomposition to get eigen vector/values

    X_pca = U.dot(X_hat.T) # Transform dataset using eigenvectors
    
    return X_pca.T, U, s


def choosing_k(lowest,highest):
    k_and_errors=[]
    for k in range(lowest,highest)
        #run k NN and calculate error
        k_and_errors.append((k,error))

    k_and_errors.sort(key=lambda x:x[1])
    best_k=k_and_errors[0][0]

def comparison(X_train,y_train,X_test,test_y,k):
        # Step 1: k-NN without PCA
    predicted_classes_wo_pca = classify_by_k_NN(X_train, y_train, X_test, k=k)
    accuracy_wo_pca = calculate_error(predicted_classes_wo_pca, test_y)
    print(f"Accuracy without PCA: {accuracy_wo_pca}")


    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1) 

    X_train_pca, U_train, s_train = pca(X_train_flat)
    X_test_pca, U_test, s_test = pca(X_test_flat)
 
    predicted_classes_pca = classify_by_k_NN(X_train_pca, y_train, X_test_pca, k=k)
    accuracy_with_pca = calculate_error(predicted_classes_pca, test_y)
    print(f"Accuracy with PCA: {accuracy_with_pca}")

    # Step 5: Compare the results
    if accuracy_with_pca > accuracy_wo_pca:
        print("PCA improves k-NN performance.")
    else:
        print("PCA does not improve k-NN performance.")