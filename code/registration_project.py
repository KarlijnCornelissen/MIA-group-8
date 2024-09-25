"""
Project code for image registration topics.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output

import cv2 as cv
from skimage.filters import gaussian

#TODO: terminate als er maar weinig verandering is in de cost function. 
#TODO: gaande weg hogere mu kiezen...
#TODO: functie schrijven voor "noice-canccelling"
#TODO: print eind transformatie
#TODO: verschil afbeelding
#TODO: pointbased



def intensity_based_registration_demo(I, Im, mu=0.0005, num_iter = 150, rigid=True, corr_metric="CC"):

    # intensity_based_registration_demo(I_path='./image_data/1_1_t1.tif', Im_path='./image_data/1_2_t1.tif',
    #                                   mu=0.0005, num_iter = 150, rigid=True, corr_metric="CC"):

    # read the fixed and moving images
    # change these in order to read different images
    # I = plt.imread(I_path)
    # Im = plt.imread(Im_path)

    if rigid:
    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
        x = np.array([0., 0., 0.])
        assert corr_metric == "CC", "combination of rigid transformation and MI correlation metric not available."
        fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=False)

    

    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
        
    
    elif rigid == False:
        x = np.array([0., 1., 1., 0., 0., 0., 0.,])
        if corr_metric =="CC":
            fun = lambda x: reg.affine_corr(I, Im, x, return_transform=False)
    
        elif corr_metric == "MI":
            fun = lambda x: reg.affine_mi(I, Im, x, return_transform=False)

    # # the learning rate
    # mu = 0.00052#0.002

    # # number of iterations
    # num_iter = 150#200

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    


    fig = plt.figure(figsize=(14,6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(Im, alpha=0.7)
    # parameters
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()

    
    # Define threshold for small changes
    threshold = 1e-5  # Small value for determining minimal change
    max_small_change_iters = 3  # The number of small change iterations needed to break
    small_change_count = 0  # Counter for consecutive small changes
    

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):
           
        # gradient ascent
        g = reg.ngradient(fun, x)
        print(g)
        
        x += g*mu
        # for visualization of the result
        if rigid:
            S, Im_t, _ = reg.rigid_corr(I, Im, x, return_transform=True)

        if rigid==False:
            if corr_metric == "MI":    
                S, Im_t, _ = reg.affine_mi(I, Im, x, return_transform=True)

            elif corr_metric == "CC":
                S, Im_t, _ = reg.affine_corr(I, Im, x, return_transform=True)

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        if k>20 and abs(similarity[k]-similarity[k-1])<threshold:
            small_change_count += 1
            #print(f"Small change detected at iteration {k}, count: {small_change_count}")
        else:
            small_change_count = 0  # Reset the count if the change is large enough

        if small_change_count >= max_small_change_iters:
            print(f"Terminating early at iteration {k} due to small change for {max_small_change_iters} consecutive iterations.")
            break
        
        display(fig)
    return Im_t, x, S


def add_noise(img_path, high=False):
    img = plt.imread(img_path)
    if high == True:
        mean = 0
        sigma = 9               # gevonden in een bron
    elif high == False:
        mean = 0
        sigma = 4.2             # gevonden in een bron

    gaussian = np.random.normal(mean, sigma, (img.shape[0],img.shape[1])) 

    noisy_image = img + gaussian

    return noisy_image

def noise_filtering(img):
    #img = plt.imread(img_path)
    #two ways, cv of skimage
    gaussian_filter = cv.GaussianBlur(img, (5,5), 0, borderType=cv.BORDER_CONSTANT)
    gaussian_filter_ski = gaussian(img, sigma=1, mode='constant', cval=0.0)

    cv.imshow("Original", img)
    cv.imshow("Gaussian 1", gaussian_filter)
    cv.imshow("Gaussian 2", gaussian_filter_ski)
    cv.waitKey(0)  # Wait for a key press
    cv.destroyAllWindows()
    return gaussian_filter, gaussian_filter_ski
    
def difference_images(img1, img2):
    im_moving, x, S = intensity_based_registration_demo(img1, img2)
    diff = img1 - im_moving
    plt.imshow(diff)
    return diff