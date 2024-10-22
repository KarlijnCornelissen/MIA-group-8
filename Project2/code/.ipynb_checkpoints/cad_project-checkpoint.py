"""
Project code for CAD topics.
"""

import numpy as np
import cad_util as util
import matplotlib.pyplot as plt
import registration as reg
import cad
import scipy
from IPython.display import display, clear_output
import scipy.io
import math
from sklearn.metrics import confusion_matrix


def nuclei_measurement():
    """
    Perform nuclei area measurement using linear regression and visualize the results.
    
    This function loads microscopy images of nuclei and their corresponding area labels.
    It trains a linear regression model to predict the area of the nuclei from the image data.
    The function visualizes the smallest and largest nuclei, compares the performanceof the linear regression model 
    using the full dataset versus a reduced dataset by plotting the predicted area vs the area, and calculates the error of both models.

    Input:
        None. The function reads data from a .mat file located at '../data/nuclei_data.mat'.
        
    Output:
        E_full : Root mean square error (RMSE) when training with the full dataset (float)
        E_reduced : Root mean square error (RMSE) when training with the reduced dataset (float)
    """
    #read in the data and dataset preparation
    fn = '../data/nuclei_data.mat'
    mat = scipy.io.loadmat(fn)
    test_images = mat["test_images"] # shape (24, 24, 3, 20730)
    test_y = mat["test_y"] # shape (20730, 1)
    training_images = mat["training_images"] # shape (24, 24, 3, 21910)
    training_y = mat["training_y"] # shape (21910, 1)

    montage_n = 300
    sort_ix = np.argsort(training_y, axis=0)
    sort_ix_low = sort_ix[:montage_n] # get the 300 smallest
    sort_ix_high = sort_ix[-montage_n:] #Get the 300 largest

    # visualize the 300 smallest and the 300 largest nuclei
    X_small = training_images[:,:,:,sort_ix_low.ravel()]
    X_large = training_images[:,:,:,sort_ix_high.ravel()]
    fig = plt.figure(figsize=(16,8))
    ax1  = fig.add_subplot(121)
    ax2  = fig.add_subplot(122)
    util.montageRGB(X_small, ax1)
    ax1.set_title('300 smallest nuclei')
    util.montageRGB(X_large, ax2)
    ax2.set_title('300 largest nuclei')

    # dataset preparation
    imageSize = training_images.shape
    
    # every pixel is a feature so the number of features is:
    # height x width x color channels
    numFeatures = imageSize[0]*imageSize[1]*imageSize[2]
    training_x = training_images.reshape(numFeatures, imageSize[3]).T.astype(float)
    test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

    ## training linear regression model
    #---------------------------------------------------------------------#
    #Training of a linear regression model for measuring the area of nuclei in microscopy images. Then, use the trained model 
    #to predict the areas of the nuclei in the test dataset.
    
    Theta = reg.ls_solve(training_x, training_y)[0]
    predicted_y = test_x.dot(Theta) 
    #---------------------------------------------------------------------#

    # visualize the results
    fig2 = plt.figure(figsize=(16,8))
    ax1  = fig2.add_subplot(121)
    line1, = ax1.plot(test_y, predicted_y, ".g", markersize=3)
    ax1.grid()
    ax1.set_xlabel('Area')
    ax1.set_ylabel('Predicted Area')
    ax1.set_title('Training with full sample')

    #training with smaller number of training samples
    #---------------------------------------------------------------------#
    # Training a model with reduced dataset size: every fourth training sample.
    reduced_training_index = np.arange(0, training_images.shape[3], 4) # Select every fourth sample from the training set
    # Create reduced training set by selecting images and labels based on the reduced index
    reduced_training_images = training_images[:, :, :, reduced_training_index]
    reduced_training_y = training_y[reduced_training_index] 
    # Determine the number of samples in the reduced training set
    num_reduced_samples = reduced_training_index.shape[0]
    # Reshape the reduced training images to a 2D array where each row is a flattened image
    reduced_training_x = reduced_training_images.reshape(numFeatures, num_reduced_samples).T.astype(float)
 
    reduced_Theta = reg.ls_solve(reduced_training_x, reduced_training_y)[0]
    reduced_predicted_y = test_x.dot(reduced_Theta)
    #---------------------------------------------------------------------#

    # visualize the results
    ax2  = fig2.add_subplot(122)
    line2, = ax2.plot(test_y, reduced_predicted_y, ".g", markersize=3)
    ax2.grid()
    ax2.set_xlabel('Area')
    ax2.set_ylabel('Predicted Area')
    ax2.set_title('Training with smaller sample')

    #---------------------------------------------------------------------#
    #Calculate the error (MEAN SQUARED ERROR)
    E_full = np.sqrt(np.mean((test_y -  predicted_y)**2))
    E_reduced = np.sqrt(np.mean((test_y - reduced_predicted_y)**2))
               
    #---------------------------------------------------------------------#

    return E_full, E_reduced

def reduce_data(training_images, training_y, percentage):
    """
    Reduce the training data the the given percentage.
    
    Input:
        training_images : The training images (numpy array)
        training_y : The training labels (numpy array)
        percentage : The percentage of the data that should be kept (int)
        
    Output:
        reduced_training_images : The reduced training images (numpy array)
        reduced_training_y : The reduced training labels (numpy array)
    """
    reduced_training_index = np.arange(0, training_images.shape[3], int(100/percentage))
    reduced_training_images = training_images[:, :, :, reduced_training_index]
    reduced_training_y = training_y[reduced_training_index]
    return reduced_training_images, reduced_training_y

def nuclei_classification():
    """
    Perform nuclei classification using logistic regression and visualize the training progress.
    
    This function loads microscopy images of nuclei and their corresponding labels, and prepares
    the data for classification. It trains a logistic regression model using gradient descent and 
    visualizes the training loss and validation loss over iterations. The function stops training 
    early if the validation loss does not decrease for a certain number of iterations.
    
    Input:
        None. The function reads data from a .mat file located at '../data/nuclei_data_classification.mat'.
        
    Output:
        Theta: The learned model parameter (numpy array)
        test_x: The reshaped and normalized test images (numpy array)
        test_y: The labels for the test images (numpy array)
        validation_x: The validation set features, reshaped and normalized validation images (numpy array)
        validation_y: The labels for the validation images (numpy array)
        training_x: The reshaped and normalized training images (numpy array)
        training_y: The labels for the training images (numpy array)
        validation_loss: The array containing validation loss for each iteration (numpy array)
        accuracy: The classification accuracy on the test set (float)
    """
    #read in the data and dataset preparation
    fn = '../data/nuclei_data_classification.mat'
    mat = scipy.io.loadmat(fn)

    test_images = mat["test_images"] # (24, 24, 3, 20730)
    test_y = mat["test_y"] # (20730, 1)
    training_images = mat["training_images"] # (24, 24, 3, 14607)
    training_y = mat["training_y"] # (14607, 1)
    validation_images = mat["validation_images"] # (24, 24, 3, 7303)
    validation_y = mat["validation_y"] # (7303, 1)

    #---------------------------------------------------------------------#
    ## reduce the training data
    # training_images, training_y = reduce_data(training_images, training_y, 0.5)
    #---------------------------------------------------------------------#

    
    training_x, validation_x, test_x = util.reshape_and_normalize(training_images, validation_images, test_images)      
    
    ## training linear regression model
    #-------------------------------------------------------------------#
    # Values for the learning rate (mu), batch size
    # (batch_size) and number of iterations (num_iterations), as well as
    # initial values for the model parameters (Theta) that will result in
    # fast training of an accurate model for this classification problem are selected.
    
    mu = 0.0001                 
    batch_size = 30            
    num_iterations = 5000       # Number of iterations for training big enough to reach convergence
    Theta = 0.02*np.random.rand(training_x.shape[1]+1, 1) # Shape (1729, 1)
    #-------------------------------------------------------------------#
    #The model is trained using the training dataset and validated using the validation dataset.

    xx = np.arange(num_iterations)
    loss = np.empty(*xx.shape)
    loss[:] = np.nan                  #array filled with NaN values, will be used to save training loss
    validation_loss = np.empty(*xx.shape)
    validation_loss[:] = np.nan       #array filled with NaN values, will be used to save validation loss
    g = np.empty(*xx.shape)
    g[:] = np.nan

    fig = plt.figure(figsize=(8,8))
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (average per sample)')
    ax2.set_title('mu = '+str(mu))
    h1, = ax2.plot(xx, loss, linewidth=2) #'Color', [0.0 0.2 0.6],
    h2, = ax2.plot(xx, validation_loss, linewidth=2) #'Color', [0.8 0.2 0.8],
    ax2.set_ylim(0, 0.7)
    ax2.set_xlim(0, num_iterations)
    ax2.grid()

    text_str2 = 'iter.: {}, loss: {:.3f}, val. loss: {:.3f}'.format(0, 0, 0)
    txt2 = ax2.text(0.3, 0.95, text_str2, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)
    #Text string to display the current iteration and loss values. This text will be updated during each iteration to reflect the progress.

    #-----------------------------------------------------------------------------------------------------------------
    different = 0 # Variable to keep track of the number of times the validation loss has not decreased
    #-----------------------------------------------------------------------------------------------------------------
    
    for k in np.arange(num_iterations):                  #for each training iteration
        # pick a batch at random
        idx = np.random.randint(training_x.shape[0], size=batch_size)

        training_x_ones = util.addones(training_x[idx,:])
        validation_x_ones = util.addones(validation_x)

        # the loss function for this particular batch
        loss_fun = lambda Theta: cad.lr_nll(training_x_ones, training_y[idx], Theta)

        # gradient descent
        # instead of the numerical gradient, we compute the gradient with
        # the analytical expression, which is much faster
        Theta_new = Theta - mu*cad.lr_agrad(training_x_ones, training_y[idx], Theta).T

        loss[k] = loss_fun(Theta_new)/batch_size
        validation_loss[k] = cad.lr_nll(validation_x_ones, validation_y, Theta_new)/validation_x.shape[0]

        output = util.addones(test_x).dot(Theta_new) # ik weet niet of dit klopt?

        # visualize the training
        h1.set_ydata(loss)
        h2.set_ydata(validation_loss)
        text_str2 = 'iter.: {}, loss: {:.3f}, val. loss={:.3f}, diff={} '.format(k, loss[k], validation_loss[k], different)
        txt2.set_text(text_str2)

        Theta = None
        Theta = np.array(Theta_new)
        Theta_new = None
        tmp = None
        
        
        #-----------------------------------------------------------------------------------------------------------------
        # Stop training if the validation loss has not decreased for 9 iterations
        if abs(validation_loss[k]-validation_loss[k-1]) < 0.001:
            different += 1
        else:
            different = 0
        if different == 9:
            break
        #-----------------------------------------------------------------------------------------------------------------
        

        # update the figure
        display(fig)
        clear_output(wait = True)
        plt.pause(.005)  
    calculate_assesments(util.addones(test_x), test_y, Theta) # calculate validation metrics

    # Display the final loss values
    ax2.set_xlim(0, k)
    display(fig)

def calculate_assesments(test_x_ones, test_y, Theta):
    """
    Calculate the recall of the nuclei classification model.
    
    Input:
        test_x_ones: The test images (numpy array) stacked with ones shape (7303, 1729)
        test_y: The true labels for the test images (numpy array) shape (7303, 1)
        Theta: The trained model parameters (numpy array) shape (1729, 1)
        
    Output:
        recall : The recall of the nuclei classification model (float)
        accuracy : The accuracy of the model (float)
        FPR : The false positive rate (float)
        TPR : The true positive rate (float)
        precision : The precision of the model (float)
        F1 : The F1 score, a harmonic mean of precision and recall (float)
        fn: The false negatives
    """
    # Predict the labels for the test set using the learned model parameters (Theta)
    predicted = np.round(test_x_ones.dot(Theta)) 
    # Ensure the predicted values are binary (0 or 1) based on a threshold of 1
    predicted = np.round(predicted >= 1).astype(int) # Round to 0 or 1
    # True labels from the test set
    true = test_y
    tn, fp, fn, tp = confusion_matrix(predicted, true).ravel() # Calculate confusion matrix
    accuracy = (tp + tn) / (tp + tn + fp + fn) # Calculate accuracy
    FPR = fp / (fp + tn) # Calculate false positive rate
    recall = tp / (tp + fn) # Calculate recall
    precision = tp / (tp + fp) # Calculate precision
    F1 = 2 * (precision * recall) / (precision + recall) # Calculate F1 score

    print('Accuracy: ', accuracy)
    print('False Positive Rate: ', FPR)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('F1 Score: ', F1)
    print("false negatives: ", fn)

def get_model_parameters_lowest_validation_loss(validation_loss, weights_list, Acc):
    """
    Retrieve the model parameters and accuracy corresponding to the lowest validation loss.

    Input:
        validation_x: The validation set features (numpy array)
        validation_y: The true labels for the validation set (numpy array)
        validation_loss: A list of validation loss values recorded during training (list)
        weights_list: A list of model weights corresponding to each validation loss value (list)
        Acc: A list of accuracy values corresponding to each validation loss value (list)
        
    Output:
        weights: The model parameters (weights) that correspond to the lowest validation loss (numpy array)
        Accuracy: The accuracy of the model associated with the lowest validation loss (float)
    """
    i = validation_loss.index(min(validation_loss))
    Accuracy = Acc[i]
    weights = weights_list[i]

    return weights, Accuracy

def get_results_testset_Neural_Network(test_x,test_y,weights):
    """
    Evaluate the performance of a neural network classifier on the test dataset.

    Input:
        weights: A dictionary containing model weights, including 'w1' (weights for the hidden layer) and 'w2' (weights for the output layer).
        test_x: The test set features (numpy array)
        test_y: The true labels for the test set (numpy array)
    
    Output:
        recall: The recall of the nuclei classification model (float)
        accuracy: The accuracy of the model (float)
        FPR: The false positive rate (float)
        TPR: The true positive rate (float)
        precision: The precision of the model (float)
        F1: The F1 score, a harmonic mean of precision and recall (float)
        fn: The false negatives
    """
    w1 = weights['w1']
    w2 = weights['w2']

    hidden = util.sigmoid(np.dot(test_x, w1))
    output = util.sigmoid(np.dot(hidden, w2))
    output = np.round(output)

    
    tn, fp, fn, tp = confusion_matrix(output, test_y).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    FPR = fp / (fp + tn) # Calculate false positive rate
    recall = tp / (tp + fn) # Calculate recall
    precision = tp / (tp + fp) # Calculate precision
    F1 = 2 * (precision * recall) / (precision + recall) # Calculate F1 score

    print('Accuracy: ', accuracy)
    print('False Positive Rate: ', FPR)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('F1 Score: ', F1)
    print("false negatives: ", fn)
 
    return accuracy, FPR, FPR, recall, precision, F1, fn

import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_assessments_knn(X_test, y_test, X_train, y_train, k):
    """
    Calculate the performance metrics for a k-NN model.

    Input:
        X_test: The test images (numpy array) shape (n_samples, n_features)
        y_test: The true labels for the test images (numpy array) shape (n_samples,)
        X_train: The training images (numpy array) shape (n_samples, n_features)
        y_train: The true labels for the training images (numpy array) shape (n_samples,)
        k: The number of neighbors to use for k-NN

    Output:
        recall: The recall of the nuclei classification model (float)
        accuracy: The accuracy of the model (float)
        FPR: The false positive rate (float)
        TPR: The true positive rate (float)
        precision: The precision of the model (float)
        F1: The F1 score, a harmonic mean of precision and recall (float)
        fn: The false negatives
    """
    #Predict the labels using k-NN
    predicted_classes = classify_by_k_NN(X_train, y_train, X_test, k=k)
    predicted = np.array(list(predicted_classes.values()))

    #Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()

    #Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)  # Calculate accuracy
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0  # Calculate false positive rate
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Calculate recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Calculate precision
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # Calculate F1 score

    #Print metrics
    print('Accuracy: ', accuracy)
    print('False Positive Rate: ', FPR)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('F1 Score: ', F1)
    print("False Negatives: ", fn)

    print('Accuracy: ', accuracy)
    print('False Positive Rate: ', FPR)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('F1 Score: ', F1)
    print("false negatives: ", fn)
 
    return accuracy, FPR, FPR, recall, precision, F1, fn

