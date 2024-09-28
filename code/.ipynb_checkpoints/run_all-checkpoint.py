import sys

from registration_project import *
import matplotlib.pyplot as plt

def run_all(corr_metric="CC"):
    """ 
    Processes T1 and T2 images for multiple patients and slices to calculate similarity measures 
    between the images after adding noise and applying filtering. The function returns a structured 
    dictionary containing the results.

    Input:
        corr_metric (str): The correlation metric to use for measuring similarity between images. 
                             - "CC": Correlation Coefficient
                             - "MI": Mutual Information

    Output:
        - Patients_dict (dict): A nested dictionary with the following structure:
            - patients (dict): Keys are patient numbers (1, 2, 3).
                - slices (dict): Keys are slice numbers (1, 2, 3).
                    - similarity_scores (list): List of similarity scores for the following cases:
                      [S_original_imgs, S_high_noise, S_low_noise, S_high_filter, S_low_filter]
    Each score corresponds to the similarity between:
    - Original T1 and T2 images
    - Noisy T1 and T2 images (with high and low noise)
    - Filtered T1 and T2 images (after applying noise filtering)
    
    The goal is to determine if applying the best transformation matrix from noisy and filtered 
    images to the original T2 leads to improved alignment. This helps to understand the impact 
    of noise on similarity measurements and whether filtering enhances registration accuracy.
        
    """
    Patients_dict = {}
    
    for patient in range(1,4):
        Patients_dict[patient]={}
        for slice in range(1,4):
            Patients_dict[patient][slice]=[]
            image_dict = {}
            # image_dict.clear
            # Load the original T1 and T2 images, for the current patient and slice
            image_dict["I"] = plt.imread(f"image_data/{patient}_{slice}_t1.tif") #deleted: MIA-group-8/code
            image_dict["Im"] = plt.imread(f"image_data/{patient}_{slice}_t2.tif")
            
            # Add noise to the images with high and low noise levels.
            image_dict["noise_high_i"] = add_noise(image_dict["I"], 'T1', True) #High noise for T1
            image_dict["noise_high_Im"] = add_noise(image_dict["Im"], 'T2', True) #High noise for T2
            image_dict["noise_low_i"] = add_noise(image_dict["I"], 'T1', False) #Low noise for T1
            image_dict["noise_low_Im"] = add_noise(image_dict["Im"], 'T2',False) #Low noise for T2

            #Voor nu bij filteren gebruiken we even zelfde sigma als eroverheen gezet is
            # Apply Gaussian filtering to the noisy images with specified sigma values.
            image_dict["filtered_high_ski"] = noise_filtering(image_dict["noise_high_i"], sigma=12.6) #High sigma filter T1
            image_dict["filtered_high_ski_Im"] = noise_filtering(image_dict["noise_high_Im"], sigma=16.2) #High sigma filter T2
            image_dict["filtered_low_ski"] = noise_filtering(image_dict["noise_low_i"],sigma=4.2) #Low sigma filter T1
            image_dict["filtered_low_ski_Im"] = noise_filtering(image_dict["noise_low_Im"], sigma=5.4) #Low sigma filter T2

            keys=list(image_dict.keys())
            if patient==3:
                mu=0.5
            else: 
                mu=0.1
            # Perform intensity-based registration to compute transformation matrices.
            for i in range(0,10,2):
                _, T, _ = intensity_based_registration_demo(image_dict[keys[i]],image_dict[keys[i+1]], initial_learning_rate=mu,
                                                            rigid=False,corr_metric=corr_metric,Plot=False)
                if corr_metric == "MI":    
                    Patients_dict[patient][slice].append(float(reg.affine_mi(image_dict["I"], image_dict["Im"], T,return_transform=False)))

                elif corr_metric == "CC":                    
                    Patients_dict[patient][slice].append(float(reg.affine_corr(image_dict["I"], image_dict["Im"], T,return_transform=False)))
            print(f"{(patient-1)*3+slice} out of {9} completed!")
    
    return Patients_dict
    

#Data_CC=run_all("")
#print(Data_CC)
#Data_MI = run_all()
#print(Data_MI)