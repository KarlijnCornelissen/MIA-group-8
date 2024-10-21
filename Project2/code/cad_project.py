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
    # TODO: Implement training of a linear regression model for measuring
    # the area of nuclei in microscopy images. Then, use the trained model
    # to predict the areas of the nuclei in the test dataset.
    
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
    # TODO: Train a model with reduced dataset size (e.g. every fourth
    # training sample).
    reduced_training_index = np.arange(0, training_images.shape[3], 4)
    reduced_training_images = training_images[:, :, :, reduced_training_index]
    reduced_training_y = training_y[reduced_training_index]  # Shape (N/4, 1)?
    num_reduced_samples = reduced_training_index.shape[0]
    reduced_training_x = reduced_training_images.reshape(numFeatures, num_reduced_samples).T.astype(float)
    ## ALS IEMAND HIER NOG NAAR KAN KIJKEN OM TE CHECKEN GRAAG##

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
    ## dataset preparation
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

    ## dataset preparation
    training_x, validation_x, test_x = util.reshape_and_normalize(training_images, validation_images, test_images)      
    
    ## training linear regression model
    #-------------------------------------------------------------------#
    # TODO: Select values for the learning rate (mu), batch size
    # (batch_size) and number of iterations (num_iterations), as well as
    # initial values for the model parameters (Theta) that will result in
    # fast training of an accurate model for this classification problem.
    # Then, train the model using the training dataset and validate it
    # using the validation dataset.
    mu = 0.0001                 # waarschijnlijk te klein
    batch_size = 30            # Batch size lijkt rond de 30 te zitten
    num_iterations = 5000       # loss is nu NAN voor eerste 150 iteraties, validation loss is de hele tijd NAN
    Theta = 0.02*np.random.rand(training_x.shape[1]+1, 1) # Shape (1729, 1)

    #-------------------------------------------------------------------#

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
    # accuracy = [] # List to store the accuracy of the model
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
        
    calculate_recall(util.addones(test_x), test_y, Theta)
    # calculate_recall(output, test_y, Theta)
    error = abs(util.addones(test_x)[k].dot(Theta) - test_y[k])/test_y.shape[0]
    # error = abs(output - test_y).sum()/test_y.shape[0]
    accuracy = (1-error)    
    # print('Error: ', error)
    print('Accuracy: ', accuracy)

    ax2.set_xlim(0, k)
    display(fig)
    return Theta, test_x, test_y, validation_x, validation_y, training_x, training_y, validation_loss, accuracy


def calculate_recall(test_x_ones, test_y, Theta):
    """
    Calculate the recall of the nuclei classification model.
    
    Input:
        test_x_ones : The test images (numpy array) stacked with ones shape (7303, 1729)
        test_y : The true labels for the test images (numpy array) shape (7303, 1)
        Theta : The trained model parameters (numpy array) shape (1729, 1)
        
    Output:
        recall : The recall of the nuclei classification model (float)
    """
    predicted = np.round(test_x_ones.dot(Theta)) 
    predicted = np.round(test_x_ones >= 1).astype(int) # Round to 0 or 1
    print(predicted[:20])
    true = test_y
    tn, fp, fn, tp = confusion_matrix(predicted, true).ravel() # Calculate confusion matrix
    accuracy = (tp + tn) / (tp + tn + fp + fn) # Calculate accuracy
    FPR = fp / (fp + tn) # Calculate false positive rate
    TPR = tp / (tp + fn) # Calculate true positive rate
    recall = tp / (tp + fn) # Calculate recall
    Precision = tp / (tp + fp) # Calculate precision
    F1 = 2 * (Precision * recall) / (Precision + recall) # Calculate F1 score

    print('Accuracy: ', accuracy)
    print('False Positive Rate: ', FPR)
    print('Recall: ', recall)
    print('Precision: ', Precision)
    print('F1 Score: ', F1)
    
    

def get_model_parameters(validation_x, validation_y, validation_loss, weights_list, Acc):
    i = validation_loss.index(min(validation_loss))
    Accuracy = Acc[i]
    weights = weights_list[i]

    return weights, Accuracy

