"""
Registration module main code.
"""

import numpy as np
from scipy import ndimage
import registration_util as util


# SECTION 1. Geometrical transformations


def identity():
    # 2D identity matrix.
    # Output:
    # T - transformation matrix

    T = np.eye(2)

    return T


def scale(sx, sy):
    # 2D scaling matrix.
    # Input:
    # sx, sy - scaling parameters
    # Output:
    # T - transformation matrix

    T = np.array([[sx,0],[0,sy]])

    return T


def rotate(phi):
    # 2D rotation matrix.
    # Input:
    # phi - rotation angle
    # Output:
    # T - transformation matrix

    #------------------------------------------------------------------#
    T = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
    #------------------------------------------------------------------#

    return T


def shear(cx, cy):
    # 2D shearing matrix.
    # Input:
    # cx - horizontal shear
    # cy - vertical shear
    # Output:
    # T - transformation matrix

    #------------------------------------------------------------------#
    T = np.array([[1,cx],[cy,1]])
    #------------------------------------------------------------------#

    return T


def reflect(rx, ry):
    # 2D reflection matrix.
    # Input:
    # rx - horizontal reflection (must have value of -1 or 1)
    # ry - vertical reflection (must have value of -1 or 1)
    # Output:
    # T - transformation matrix

    allowed = [-1, 1]
    if rx not in allowed or ry not in allowed:
        #raise ValueError(f"Invalid input parameters for reflect: rx={rx}, ry={ry}. Allowed values are -1 or 1.")
        T = 'Invalid input parameter'
        #return T
    
    T = np.array([[rx, 0],
                  [0, ry]])
        

    #------------------------------------------------------------------#
    T=np.array([[rx,0],[0,ry]])
    #------------------------------------------------------------------#

    return T


# SECTION 2. Image transformation and least squares fitting


def image_transform(I, Th,  output_shape=None):
    # Image transformation by inverse mapping.
    # Input:
    # I - image to be transformed
    # Th - homogeneous transformation matrix
    # output_shape - size of the output image (default is same size as input)
    # Output:
    # It - transformed image
	# Xt - remapped coordinates
    # we want double precision for the interpolation, but we want the
    # output to have the same data type as the input - so, we will
    # convert to double and remember the original input type

    input_type = type(I)

    # default output size is same as input
    if output_shape is None:
        output_shape = I.shape

    # spatial coordinates of the transformed image
    x = np.arange(0, output_shape[1])
    y = np.arange(0, output_shape[0])
    xx, yy = np.meshgrid(x, y)

    # convert to a 2-by-p matrix (p is the number of pixels)
    X = np.concatenate((xx.reshape((1, xx.size)), yy.reshape((1, yy.size))))
    # convert to homogeneous coordinates
    Xh = util.c2h(X)

    #------------------------------------------------------------------#
    det = np.linalg.det(Th)
    if det == 0:
        raise ValueError("the transformation matrix is not invertible")
    else:
        Th_inv=np.linalg.inv(Th)
        Xt_h = Th_inv.dot(Xh)  # Apply the inverse transformation
        Xt = Xt_h[:2, :]       # Extract x and y coordinates
        Xt = Xt.reshape((2, xx.shape[0], xx.shape[1]))  # Reshape to match (2, height, width)
    
    #------------------------------------------------------------------#

    It = ndimage.map_coordinates(I, [Xt[1,:], Xt[0,:]], order=1, mode='constant').reshape(output_shape)

    return It, Xt


def ls_solve(A, b):
    # Least-squares solution to a linear system of equations.
    # Input:
    # A - matrix of known coefficients
    # b - vector of known constant term
    # Output:
    # w - least-squares solution to the system of equations
    # E - squared error for the optimal solution

    #------------------------------------------------------------------#
    # Compute least squares solution using normal equations
    AtA = np.transpose(A).dot(A)  # A^T * A
    Atb = np.transpose(A).dot(b)  # A^T * b
    
    # Solve the system AtA * w = Atb to find w
    w = np.linalg.solve(AtA, Atb)
    

    #------------------------------------------------------------------#

    # compute the error
    E = np.transpose(A.dot(w) - b).dot(A.dot(w) - b)

    return w, E


def ls_affine(X, Xm):
    # Least-squares fitting of an affine transformation.
    # Input:
    # X - Points in the fixed image
    # Xm - Corresponding points in the moving image
    # Output:
    # T - affine transformation in homogeneous form.

    A = np.transpose(Xm)
    
    #------------------------------------------------------------------#
    # TODO: Implement least-squares fitting of an affine transformation.
    # Use the ls_solve() function that you have previously implemented.
    
    b1 = np.transpose(np.array([X[0,:]]))
    b2 = np.transpose(np.array([X[1,:]]))
    w1, E1 = ls_solve(A,b1)
    w2, E2 = ls_solve(A,b2)
	
    #linking w1 and w2 together, and adding 0,0,1 at the end, added by columns so from left to right w1,w2,0
    T = np.concatenate((w1,w2,[[0],[0],[1]]), axis=1)
    #axis = 0 means rows, arrays on top of each other
    #axis = 1 meas columns, arrays side by side
    T = np.transpose(T)
    

    
    #------------------------------------------------------------------#

    return T


# SECTION 3. Image simmilarity metrics


def correlation(I, J):
    # Compute the normalized cross-correlation between two images.
    # Input:
    # I, J - input images
    # Output:
    # CC - normalized cross-correlation
    # it's always good to do some parameter checks

    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")

    u = I.reshape((I.shape[0]*I.shape[1],1))
    v = J.reshape((J.shape[0]*J.shape[1],1))

    # subtract the mean
    u = u - u.mean(keepdims=True) #dimensies blijven hetzelfde, dus wordt vector - vector(allemaal zelfde getal (gemiddelde))
    v = v - v.mean(keepdims=True)



    #------------------------------------------------------------------#
    # TODO: Implement the computation of the normalized cross-correlation.
    # This can be done with a single line of code, but you can use for-loops instead.
    teller = np.transpose(u).dot(v)
    noemer= np.sqrt(np.transpose(u).dot(u)) * np.sqrt(np.transpose(v).dot(v))
    CC= teller/noemer
    #------------------------------------------------------------------#

    return CC


def joint_histogram(I, J, num_bins=16, minmax_range=None):
    # Compute the joint histogram of two signals.
    # Input:
    # I, J - input images
    # num_bins: number of bins of the joint histogram (default: 16)
    # range - range of the values of the signals (defaul: min and max
    # of the inputs)
    # Output:
    # p - joint histogram

    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")

    # make sure the inputs are column-vectors of type double (highest
    # precision)
    I = I.reshape((I.shape[0]*I.shape[1],1)).astype(float) 
    J = J.reshape((J.shape[0]*J.shape[1],1)).astype(float)
    #*computes the total number of pixels in the image, reshapes the 2D image array in 2D column vector, each pixel in a separate row, only one column and at
    #last conversion to float 

    # if the range is not specified use the min and max values of the
    # inputs
    if minmax_range is None:
        minmax_range = np.array([min(min(I),min(J)), max(max(I),max(J))])

    # this will normalize the inputs to the [0 1] range
    I = (I-minmax_range[0]) / (minmax_range[1]-minmax_range[0])
    J = (J-minmax_range[0]) / (minmax_range[1]-minmax_range[0])

    # and this will make them integers in the [0 (num_bins-1)] range
    I = np.round(I*(num_bins-1)).astype(int)
    J = np.round(J*(num_bins-1)).astype(int)

    n = I.shape[0]
    hist_size = np.array([num_bins,num_bins])

    # initialize the joint histogram to all zeros
    p = np.zeros(hist_size)

    for k in range(n):
        p[I[k], J[k]] = p[I[k], J[k]] + 1

    #------------------------------------------------------------------#
    # TODO: At this point, p contains the counts of cooccuring
    # intensities in the two images. You need to implement one final
    # step to make p take the form of a probability mass function
    # (p.m.f.).

    #for p.m.f the histogram needs to be normalized:
    p=p/n # n is the same as np.sum(p)
    #------------------------------------------------------------------#

    return p


def mutual_information(p):
    # Compute the mutual information from a joint histogram.
    # Input:
    # p - joint histogram
    # Output:
    # MI - mutual information in nat units
    # a very small positive number

    EPSILON = 10e-10

    # add a small positive number to the joint histogram to avoid
    # numerical problems (such as division by zero)
    p += EPSILON
    MI = 0

    # we can compute the marginal histograms from the joint histogram
    p_I = np.sum(p, axis=1)
    p_I = p_I.reshape(-1, 1)
    p_J = np.sum(p, axis=0)
    p_J = p_J.reshape(1, -1)

    #------------------------------------------------------------------#
    # TODO: Implement the computation of the mutual information from p,
    # p_I and p_J. This can be done with a single line of code, but you
    # can use a for-loop instead.
    # HINT: p_I is a column-vector and p_J is a row-vector so their
    # product is a matrix. You can also use the sum() function here.

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            MI += p[i,j]*np.log(p[i,j]/(p_I[i,0]*p_J[0,j]))
            
                                
    #------------------------------------------------------------------#

    return MI


def mutual_information_e(p):
    # Compute the mutual information from a joint histogram.
    # Alternative implementation via computation of entropy.
    # Input:
    # p - joint histogram
    # Output:
    # MI - mutual information in nat units
    # a very small positive number

    EPSILON = 10e-10

    # add a small positive number to the joint histogram to avoid
    # numerical problems (such as division by zero)
    p += EPSILON

    # we can compute the marginal histograms from the joint histogram
    p_I = np.sum(p, axis=1)
    p_I = p_I.reshape(-1, 1)
    p_J = np.sum(p, axis=0)
    p_J = p_J.reshape(1, -1)

    #------------------------------------------------------------------#
    # TODO: Implement the computation of the mutual information via
    # computation of entropy.
    p_Ie = -np.sum(p_I * np.log(p_I))
    p_Je = -np.sum(p_J * np.log(p_J))
    p_Jointe = -np.sum(p*np.log(p))

    MI = p_Ie + p_Je - p_Jointe

    #n=p.shape[0]
    #H_IJ_joint = 0
    #H_I = 0
    #H_J = 0
    #for i in range(n):
    #    H_I -= p_I[i][0] * np.log(p_I[i][0])
    #    for j in range(n):
    #        H_IJ_joint -= p[i][j] * np.log(p[i][j])
    #        if i==0:
    #            H_J -= p_J[0][j] * np.log(p_J[0][j])
    #print(H_I,H_J,H_IJ_joint)
    #MI = H_I + H_J - H_IJ_joint

    #------------------------------------------------------------------#

    return MI


# SECTION 4. Towards intensity-based image registration


def ngradient(fun, x, h=1e-3):
    # Computes the derivative of a function with numerical differentiation.
    # Input:
    # fun - function for which the gradient is computed
    # x - vector of parameter values at which to compute the gradient
    # h - a small positive number used in the finite difference formula
    # Output:
    # g - vector of partial derivatives (gradient) of fun

    g = np.zeros_like(x)

    #------------------------------------------------------------------#
    # TODO: Implement the  computation of the partial derivatives of
    # the function at x with numerical differentiation.
    # g[k] should store the partial derivative w.r.t. the k-th parameter

    #geen y meegegeven aan de functie??
    #loop over parameter values at which you compute gradient:
    # for k in range(len(x)):
    #     #making a copy so you don't alter vector x
    #     xi=np.copy(x)

    #     g[k] = (fun(xi[k]+h/2)-fun(xi[k]-h/2))/h

    for i in range(len(x)):
        x_temp = np.copy(x)
        x_temp[i] += 0.5*h
        f_plus = fun(x_temp)

        x_temp[i] -= h
        f_minus = fun(x_temp)
        #x=x[i]
        #sym_der = (fun(x+1/2*h)-fun(x-1/2*h))/h
        sym_der = (f_plus - f_minus)/h
        g[i] = sym_der
    #------------------------------------------------------------------#

    return g


def rigid_corr(I, Im, x, return_transform=True):
    # Computes normalized cross-correlation between a fixed and
    # a moving image transformed with a rigid transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the rotation angle and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)

    SCALING = 100

    # the first element is the rotation angle
    T = rotate(x[0]) #rotation matrix based on the angle of first element in x

    # the remaining two element are the translation
    #
    # the gradient ascent/descent method work best when all parameters
    # of the function have approximately the same range of values
    # this is  not the case for the parameters of rigid registration
    # where the transformation matrix usually takes  much smaller
    # values compared to the translation vector this is why we pass a
    # scaled down version of the translation vector to this function
    # and then scale it up when computing the transformation matrix
    Th = util.t2h(T, x[1:]*SCALING) #Convert a 2D transformation matrix to homogeneous form, combine T(matrix) and the second and third element (vector) of                                        x, which will be scalled

    # transform the moving image
    Im_t, Xt = image_transform(Im, Th)  #It - transformed image and Xt - remapped coordinates

    # compute the similarity between the fixed and transformed
    # moving image
    C = correlation(I, Im_t)

    if return_transform:
        return C, Im_t, Th
    else:
        return C


def affine_corr(I, Im, x, return_transform=True):
    # Computes normalized cross-corrleation between a fixed and
    # a moving image transformed with an affine transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the roation angle, the second and third are the
    #     scaling parameters, the fourth and fifth are the
    #     shearing parameters and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)
    
    NUM_BINS = 64
    SCALING = 100

    #------------------------------------------------------------------#
    # TODO: Implement the missing functionality
    rot = x[0]    # Rotation angle in radians
    s_x = x[1]    # Scaling in x-direction
    s_y = x[2]    # Scaling in y-direction
    h_x = x[3]    # Shearing in x-direction
    h_y = x[4]    # Shearing in y-direction
    t_x = x[5]    # Translation in x-direction
    t_y = x[6]    # Translation in y-direction

    # Construct the affine transformation matrix for 2D
    T = rotate(rot)                                 #rotation
    T2 = np.array([s_x,0],[0,s_y]).dot(T1)          #scale
    T3 = np.array([1,h_x],[h_y,1]).dot(T2)          #shear
    Th = util.t2h(T3, [t_x, t_y]*SCALING)           #translation

    #transform moving image
    Im_t, Xt = image_transform(Im, Th)              #It - transformed image and Xt - remapped coordinates

    C = correlation(I, Im_t)
    #------------------------------------------------------------------#

    if return_transform:
        return C, Im_t, Th
    else:
        return C


def affine_mi(I, Im, x, return_transform=True):
    # Computes mutual information between a fixed and
    # a moving image transformed with an affine transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the rotation angle, the second and third are the
    #     scaling parameters, the fourth and fifth are the
    #     shearing parameters and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)

    NUM_BINS = 64
    SCALING = 100
    
    #------------------------------------------------------------------#
    T1 = rotate(x[0])                                 #rotate
    T2 = np.array([x[1],0],[0,x[2]]).dot(T1)          #scale
    T3 = np.array([1,x[3]],[x[4],1]).dot(T2)          #shear
    Th = util.t2h(T3, x[5:]*SCALING)                  #translation
    
    # transform the moving image
    Im_t, Xt = image_transform(Im, Th)
    
    rot_scal_shear = matrix[:2, :2]  # 2x2 matrix for rotation, scaling, and shearing
    translation = matrix[:2, 2]         # Translation vector

    Im_t, Xt = image_transform(Im, Th)     #It - transformed image and Xt - remapped coordinates
   

    p = joint_histogram(I, Im_t, num_bins=NUM_BINS, minmax_range=None)
    C=mutual_information(p)
    #------------------------------------------------------------------#

    if return_transform:
        return C, Im_t, Th
    else:
        return C
