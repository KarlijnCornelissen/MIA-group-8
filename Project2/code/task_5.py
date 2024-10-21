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

def k_means(X_train, y_train, X_test, k=3): 
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
    
def classify_by_k_means(X_train,y_train,X_test,k=3):
    predicted_classes={}
    for i in range(len(X_test)):
        predicted_class = k_means(X_train, y_train, X_test[i], k=k)
        predicted_classes[X_test[i]]=predicted_class

     return predicted_classes

