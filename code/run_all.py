import sys

from registration_project import *
import matplotlib.pyplot as plt
import pandas as pd

def run_all(corr_metric="CC"):
    """ returns a dictionary with the following format:
    patients_dict
        patients(dict)
            slices(list): similarity of I with IM, with different Transformation 
                            metricces that are generated in several ways
                list with the format: [S_original_imgs, S_high_noise, S_Low_noise, S_high_filter, S_low_filter]

    """
    Patients_dict = {}
    
    for patient in range(1,4):
        Patients_dict[patient]={}
        for slice in range(1,4):
            Patients_dict[patient][slice]=[]
            image_dict = {}
            # image_dict.clear
            image_dict["I"] = plt.imread(f"MIA-group-8/code/image_data/{patient}_{slice}_T1.tif")
            image_dict["Im"] = plt.imread(f"MIA-group-8/code/image_data/{patient}_{slice}_T2.tif")

            image_dict["noise_high_i"] = add_noise(image_dict["I"], 'T1', True) #T1 met veel noise
            image_dict["noise_high_Im"] = add_noise(image_dict["Im"], 'T2', True) #T2 met veel noise
            image_dict["noise_low_i"] = add_noise(image_dict["I"], 'T1', False) #T1 met weinig noise
            image_dict["noise_low_Im"] = add_noise(image_dict["Im"], 'T2',False) #T2 met weinig noise 

            #Voor nu bij filteren gebruiken we even zelfde sigma als eroverheen gezet is
            image_dict["filtered_high_ski"] = noise_filtering(image_dict["noise_high_i"], sigma=16.2) #T1 hoog gefilterd #16.2
            image_dict["filtered_high_ski_Im"] = noise_filtering(image_dict["noise_high_Im"], sigma=16.2) #T2 hoog gefilterd
            image_dict["filtered_low_ski"] = noise_filtering(image_dict["noise_low_i"],sigma=4.2) #T1 laag gefilterd
            image_dict["filtered_low_ski_Im"] = noise_filtering(image_dict["noise_low_Im"], sigma=4.2) #T2 laag gefilterd 

            keys=list(image_dict.keys())
            if patient==3:
                mu=0.05
            else: 
                mu=0.01
            for i in range(0,10,2):
                _, T, _ = intensity_based_registration_demo(image_dict[keys[i]],image_dict[keys[i+1]], initial_learning_rate=mu,
                                                            rigid=False,corr_metric=corr_metric,Plot=False)
                if corr_metric == "MI":    
                    Patients_dict[patient][slice].append(float(reg.affine_mi(image_dict["I"], image_dict["Im"], T,return_transform=False)))

                elif corr_metric == "CC":                    
                    Patients_dict[patient][slice].append(float(reg.affine_corr(image_dict["I"], image_dict["Im"], T,return_transform=False)))
            print(f"{(patient-1)*3+slice} out of {9} completed!")
    
    return Patients_dict
    

def save_data_to_csv(method="CC"):

    Data=run_all(method)
    print(Data)
    dict_of_df = {k: pd.DataFrame.from_dict(v,orient="index") for k,v in Data.items()}
    df = pd.concat(dict_of_df, axis=0)
    print(df)

    df.to_csv("MIA-group-8\code\{method}_data.csv")
    

df_CC=pd.read_csv("MIA-group-8\code\CC_data.csv",index_col=[0,1])
print(df_CC)
