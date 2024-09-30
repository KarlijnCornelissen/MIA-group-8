import sys

from registration_project import *
import matplotlib.pyplot as plt
import pandas as pd

from pytictoc import TicToc
t = TicToc() #create instance of class

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
            t.tic() #Start timer
            Patients_dict[patient][slice]=[]
            image_dict = {}
            # Load the original T1 and T2 images, for the current patient and slice
            image_dict["I"] = plt.imread(f"image_data/{patient}_{slice}_t1.tif") #added again: MIA-group-8/code
            image_dict["Im"] = plt.imread(f"image_data/{patient}_{slice}_t2.tif")
            
            # Add noise to the images with high and low noise levels.
            image_dict["noise_high_i"] = add_noise(image_dict["I"], 'T1', True) #High noise for T1
            image_dict["noise_high_Im"] = add_noise(image_dict["Im"], 'T2', True) #High noise for T2
            image_dict["noise_low_i"] = add_noise(image_dict["I"], 'T1', False) #Low noise for T1
            image_dict["noise_low_Im"] = add_noise(image_dict["Im"], 'T2',False) #Low noise for T2

           
            # Apply Gaussian filtering to the noisy images with specified sigma values.
            image_dict["filtered_high_ski"] = noise_filtering(image_dict["noise_high_i"], sigma=12.6) #High sigma filter T1
            image_dict["filtered_high_ski_Im"] = noise_filtering(image_dict["noise_high_Im"], sigma=16.2) #High sigma filter T2
            image_dict["filtered_low_ski"] = noise_filtering(image_dict["noise_low_i"],sigma=4.2) #Low sigma filter T1
            image_dict["filtered_low_ski_Im"] = noise_filtering(image_dict["noise_low_Im"], sigma=5.4) #Low sigma filter T2

            keys=list(image_dict.keys())
            if patient==3:
                mu=0.05
            else: 
                mu=0.01
            
            for i in range(0,10,2):
                # Perform intensity-based registration to compute transformation matrices:
                _, T, _ = intensity_based_registration_demo(image_dict[keys[i]],image_dict[keys[i+1]], initial_learning_rate=mu,
                                                            rigid=False,corr_metric=corr_metric,Plot=False)
                
                #choose the similarity metric that was given:
                    #apply transformation to original images, to obtain the Similarity values. 
                    #place these values immediately in the patiens_dict dictionary.
                if corr_metric == "MI":    
                    Patients_dict[patient][slice].append(float(reg.affine_mi(image_dict["I"], image_dict["Im"], T,return_transform=False)))

                elif corr_metric == "CC":                    
                    Patients_dict[patient][slice].append(float(reg.affine_corr(image_dict["I"], image_dict["Im"], T,return_transform=False)))

            #keep track of progress:
            t.toc() #Time elapsed since t.tic()
            print(f"{(patient-1)*3+slice} out of {9} completed!")
    
    return Patients_dict
    

def save_data_to_csv(method="CC"):
    """ run the function run_all, and save the result to a csv file.
    
    you only have to run this function once, after that, you can collect the data using:
    "...
    df_CC=pd.read_csv("MIA-group-8\code\CC_data.csv",index_col=[0,1])
    df_MI = pd.read_csv("MIA-group-8\code\MI_data.csv",index_col=[0,1])
    print(df_CC)
    print(df_MI)
    ..."

    args:
        - method(str): CC or MI
    """

    Data=run_all(method)
    print(Data)
    dict_of_df = {k: pd.DataFrame.from_dict(v,orient="index") for k,v in Data.items()}
    df = pd.concat(dict_of_df, axis=0)
    print(df)

    df.to_csv(f"MIA-group-8\code\{method}_data.csv")
    
# save_data_to_csv("CC")
save_data_to_csv("MI")

# df_CC=pd.read_csv("MIA-group-8\code\CC_data.csv",index_col=[0,1])
# df_MI = pd.read_csv("MIA-group-8\code\MI_data.csv",index_col=[0,1])
# print(df_CC)
# print(df_MI)
