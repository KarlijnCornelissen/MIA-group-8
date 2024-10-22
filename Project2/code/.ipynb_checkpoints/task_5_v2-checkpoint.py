import numpy as np
import scipy.io
import cad_util as util  # Assuming you have a util module with the reshape_and_normalize function

def loading_data():
    """
    Loads training and testing data from a .mat file.
    
    Input: None
    
    Output:
    - test_images: np.ndarray
        The test images array (24, 24, 3, 20730).
    - test_y: np.ndarray
        The labels for the test images (20730, 1).
    - training_images: np.ndarray
        The training images array (24, 24, 3, 21910).
    - training_y: np.ndarray
        The labels for the training images (21910, 1).
    """
    fn = '../data/nuclei_data.mat'
    mat = scipy.io.loadmat(fn)
    test_images = mat["test_images"]
    test_y = mat["test_y"]
    training_images = mat["training_images"]
    training_y = mat["training_y"]
    return test_images, test_y, training_images, training_y

def distance(x1, x2):
    """
    Computes the Euclidean distance between two data points.
    
    Input:
    - x1: np.ndarray
        The first data point (image).
    - x2: np.ndarray
        The second data point (image).
    
    Output:
    - euclidean_distance: float
        The Euclidean distance between the two input data points.
    """
    euclidean_distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return euclidean_distance

def choosing_k(X_train, y_train, X_test, test_y, lowest, highest):
    """
    Determines the best 'k' value for k-NN between a range of values.
    
    Input:
    - X_train: np.ndarray
        The training data (flattened).
    - y_train: np.ndarray
        The labels for the training data.
    - X_test: np.ndarray
        The test data (flattened).
    - test_y: np.ndarray
        The true labels for the test data.
    - lowest: int
        The lower bound of the range for 'k'.
    - highest: int
        The upper bound of the range for 'k'.
    
    Output:
    - best_k: int
        The value of 'k' that gives the lowest error rate.
    """
    k_and_errors = []
    for k in range(lowest, highest + 1):
        predicted_classes = classify_by_k_NN(X_train, y_train, X_test, k=k)
        accuracy = calculate_error(predicted_classes, test_y)
        error_rate = 1 - accuracy
        k_and_errors.append((k, error_rate))

    k_and_errors.sort(key=lambda x: x[1])
    best_k = k_and_errors[0][0]  # Return the k with the lowest error rate
    print('The chosen k is', best_k)
    return best_k

def k_NN(X_train, y_train, X_test, k=None):
    """
    Input: 
    - X_train: Training dataset (flattened if needed)
    - y_train: Training labels
    - X_test: Test image (flattened if needed)
    - k: Number of nearest neighbors

    Output: 
    - predicted_class: Predicted category for the given test image
    """
    if k is None:
        raise ValueError("k must be provided. Use choosing_k to determine the best k.")

    all_distances = []
    # Flatten X_test to ensure it's compatible with X_train
    X_test_flat = X_test.flatten()  # Flatten test image
    
    for i in range(len(X_train)):
        # Flatten the training image
        X_train_flat = X_train[i].flatten()
        
        # Calculate distance between flattened images
        dist = distance(X_train_flat, X_test_flat)
        all_distances.append((dist, y_train[i]))  # Append a tuple with distance and label

    # Sort by distance and get k nearest neighbors
    all_distances.sort(key=lambda x: x[0])
    neighbours = all_distances[:k]
    
    # Get the most frequent class among neighbors
    neighbour_classes = [neigh[1] for neigh in neighbours]
    predicted_class = max(set(neighbour_classes), key=neighbour_classes.count)
    return predicted_class

def classify_by_k_NN(X_train, y_train, X_test, k=None):
    """
    Classifies all test data samples using k-NN.
    Input:
    - X_train (the training data), 
    - y_train (the labels for the training data)
    - X_test: (the test data)
    - test_y (the true labels for the test data)
    - k: int
        The number of nearest neighbors to consider.
    
    Output:
    - predicted_classes: dict
        A dictionary where keys are test sample indices and values are predicted class labels.
    """
    if k is None:
        raise ValueError("k must be provided. Use choosing_k to determine the best k.")

    predicted_classes = {}
    for i in range(len(X_test)):
        predicted_class = k_NN(X_train, y_train, X_test[i], k=k)
        predicted_classes[i] = predicted_class
    return predicted_classes

def pca(X):
    """
    Performs Principal Component Analysis (PCA) on the input data.
    
    Input:
    - X: np.ndarray
        The input data matrix (n_samples, n_features).
    
    Output:
    - X_pca: np.ndarray
        The transformed data in the PCA space.
    - U: np.ndarray
        The matrix of principal components (eigenvectors).
    - s: np.ndarray
        The eigenvalues corresponding to the principal components.
    """
    n_samples = X.shape[0]
    X_mean = np.mean(X, axis=0)
    X_hat = X - X_mean  # Center data
    sigma_hat = (1 / (n_samples - 1)) * X_hat.T.dot(X_hat)
    U, s, V = np.linalg.svd(sigma_hat)
    X_pca = U.dot(X_hat.T)
    return X_pca.T, U, s

def comparison(X_train, y_train, X_test, test_y, k=None):
    """
    Compares the performance of k-NN with and without PCA.
    
    Input:
    - X_train (the training data), 
    - y_train (the labels for the training data)
    - X_test: (the test data)
    - test_y (the true labels for the test data)
    - k: The number of nearest neighbors to use for k-NN 
    
    Output: Prints the accuracies with and without PCA.
    """
    predicted_classes_wo_pca = classify_by_k_NN(X_train, y_train, X_test, k=k)
    accuracy_wo_pca = calculate_error(predicted_classes_wo_pca, test_y)
    print(f"Accuracy without PCA: {accuracy_wo_pca}")

    # Flatten the training and test images
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    X_train_pca, _, _ = pca(X_train_flat)
    X_test_pca, _, _ = pca(X_test_flat)

    predicted_classes_pca = classify_by_k_NN(X_train_pca, y_train, X_test_pca, k=k)
    accuracy_with_pca = calculate_error(predicted_classes_pca, test_y)
    print(f"Accuracy with PCA: {accuracy_with_pca}")

    if accuracy_with_pca > accuracy_wo_pca:
        print("PCA improves k-NN performance.")
    else:
        print("PCA does not improve k-NN performance.")

def main():
    """
    Main function to run the k-NN comparison with and without PCA.
    """
    test_images, test_y, training_images, training_y = loading_data()

    # Use the reshape_and_normalize utility function to preprocess the images
    training_images, validation_images, test_images = util.reshape_and_normalize(training_images, test_images, test_images)

    # Flatten the training and test images before using them
    X_train_flat = training_images.reshape(training_images.shape[0], -1)
    X_test_flat = test_images.reshape(test_images.shape[0], -1)

    best_k = choosing_k(X_train_flat, training_y, X_test_flat, test_y, 1, 8)
    print(f"Best k chosen: {best_k}")

    comparison(training_images, training_y, test_images, test_y, best_k)

main()
