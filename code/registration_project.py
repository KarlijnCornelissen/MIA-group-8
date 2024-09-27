"""
Project code for image registration topics.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output
from skimage.filters import gaussian
import math

#<<<<<<< HEAD
#>>>>>>> 7ebe69c51067b9879b520804769b11be6aeed489

#TODO: variabelen en functie namen fixen
#TODO: pointbased




def lr_exp_decay(initial_learning_rate, itteration):
    k = 0.3
    return initial_learning_rate * math.exp(-k * itteration)

def intensity_based_registration_demo(I, Im, initial_learning_rate=0.01, num_iter = 150, rigid=True, corr_metric="CC", Plot=True): #mu=0.0005

    if rigid:
    # initial values for the parameters
        x = np.array([0., 0., 0.])
        assert corr_metric == "CC", "combination of rigid transformation and MI correlation metric not available."
        fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=False)

    elif rigid == False:
        x = np.array([0., 1., 1., 0., 0., 0., 0.,])
        if corr_metric =="CC":
            fun = lambda x: reg.affine_corr(I, Im, x, return_transform=False)
    
        elif corr_metric == "MI":
            fun = lambda x: reg.affine_mi(I, Im, x, return_transform=False)

    

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    

    if Plot == True:

        fig = plt.figure(figsize=(14,6))

        # fixed and moving image, and parameters
        ax1 = fig.add_subplot(121)

        # fixed image
        im1 = ax1.imshow(I)
        # moving image
        im2 = ax1.imshow(Im, alpha=0.7)
        # parameters
        txt = ax1.text(0, 0.95,
            np.array2string(x, precision=5, floatmode='fixed'),
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
            transform=ax1.transAxes)
        
        #more parameters(mu and S):
        txt2 = ax1.text(0, 0," ",
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
            transform=ax1.transAxes)



        # 'learning' curve
        ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

        learning_curve, = ax2.plot(iterations, similarity, lw=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Similarity')
        ax2.grid()

    
    # Define threshold for small changes
    threshold = 5e-4  # Small value for determining minimal change
    max_small_change_iters = 8  # The number of small change iterations needed to break
    small_change_count = 0  # Counter for consecutive small changes
    

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):
           
        # gradient ascent
        g = reg.ngradient(fun, x)
        #print(g)
        
        mu = lr_exp_decay(initial_learning_rate, itteration=k)
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

        similarity[k] = S
        if Plot ==True:
            
            # update moving image and parameters
            im2.set_data(Im_t)
            txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

            #display mu and S parameters:
            txt2.set_text(f"mu={mu} and S = {float(S)}")

            # update 'learning' curve
        
            learning_curve.set_ydata(similarity)
            ax2.set_xlim(xmin=1, xmax=k)
            display(fig)
        #terminate early:
        if k>20 and abs(similarity[k]-similarity[k-1])<threshold:        
            small_change_count += 1
            #print(f"Small change detected at iteration {k}, count: {small_change_count}")
        else:
            small_change_count = 0  # Reset the count if the change is large enough

        if small_change_count >= max_small_change_iters:
            #print(f"Terminating early at iteration {k} due to small change for {max_small_change_iters} consecutive iterations.")
            break

    return Im_t, x, similarity[k]
    	

def add_noise(img, T, high=False):
    """This function adds noise to a given image. 
    params:
        -img(array): The image without noise
        -T(string): T1 or T2
        -high(Boolean): determines amount of noise added to the image
        
    Returns:
        noisy_image(array): the given image with added noise.
    """
    #img = plt.imread(img_path)
    mean = 0
    if T == "T1":
        if high == True:
            sigma = 12.6            # gevonden in een bron
        elif high == False:
            sigma = 4.2             # gevonden in een bron
    elif T == "T2":
        if high == True:
            sigma = 16.2            # gevonden in een bron
        elif high == False:
            sigma = 5.4             # gevonden in een bron
    else:
        print("Invalid T value")

    gaussian = np.random.normal(mean, sigma, (img.shape[0],img.shape[1])) 

    noisy_image = img + gaussian

    return noisy_image

def noise_filtering(img,sigma=1):
    """	Function to filter noise from an image using a Gaussian filter."""	
    return  gaussian(img, sigma=sigma, mode='constant', cval=0.0)
    
def difference_images(img1, img2):
    im_moving, x, S = intensity_based_registration_demo(img1, img2)
    diff = img1 - im_moving
    plt.imshow(diff)
    return diff