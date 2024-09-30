#Data analysis 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.close()
# Creating normalized version:
def normalized_data(title):
    df=pd.read_csv(f"MIA-group-8\code\{title}.csv", index_col=[0,1])
    df.reset_index(drop=True)

    index_names= [f"image: {i}" for i in range(1,len(df.iloc[:,0])+1)]
    df["Image number"]=index_names
    
    df.set_index("Image number",inplace=True)
    df.columns = ["Original images", "High noise level", "Low noise level", "Large filter", "Small filter"]
    for i in index_names:
        for j in df.columns:
            df.loc[i]=(df.loc[i])/(df.loc[i,"Original images"])

    
    normalized_title = f"{title}_normalized"
    file_path = f"MIA-group-8/code/{normalized_title}.csv"

    # Save the DataFrame to a CSV file
    df.to_csv(file_path)
    return df


def make_scatterplot(df):
    sns.set_theme(style="ticks", palette="pastel")
    #print(pd.melt(df))
    variables = df.columns
    sns.scatterplot(data=df, x=df.index, y=df.columns[0:5], hue=variables[1:5])
    plt.show()

def plot_results(df_CC, df_MI):
    sns.set_theme(style="ticks", palette="pastel")
    

    
    # Draw a nested boxplot to show bills by day and time
    # sns.catplot(x="variable", y="value",
    #             data=pd.melt(df), kind="violin", hue="variable")
    # plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    
    
    #CC:
    df_CC = pd.melt(df_CC)
    df_CC.columns = ["Image type", "Normalized Similarity" ]

    hue_order = df_CC['Image type'].unique()
    


    sns.stripplot(ax=ax1, 
        data=df_CC, x="Image type", y="Normalized Similarity", hue=None, hue_order=hue_order,
        )
    sns.pointplot(ax=ax1,
        data=df_CC, x="Image type", y="Normalized Similarity", hue="Image type", hue_order=hue_order,
        dodge=.8 - .8 / 3, palette="dark", errorbar=None,
        markers="d", markersize=4, linestyle="none",
        )
    # MI:
    ax2 = plt.subplot(212)
    df_MI = pd.melt(df_MI)
    df_MI.columns = ["Image type", "Normalized Similarity" ]

    sns.stripplot(ax=ax2,
        data=df_MI, x="Image type", y="Normalized Similarity", hue=None, hue_order=hue_order,
        )
    sns.pointplot(ax=ax2,
        data=df_MI, x="Image type", y="Normalized Similarity", hue="Image type",hue_order=hue_order,
        dodge=.8 - .8 / 3, palette="dark", errorbar=None,
        markers="d", markersize=4, linestyle="none",
        ) 
    ax1.legend_.remove()
    ax2.legend_.remove()
    sns.despine(ax=ax1, left=False, bottom=True)
    sns.despine(ax=ax2, left=False, bottom=False)
    ax1.tick_params(bottom=False)
    ax1.set_xlabel('')
    ax2.set_xlabel('')

    ax1.set_ylabel("Normalized similarity using\nCC as a similarity metric")
    ax2.set_ylabel("Normalized similarity using\nMI as a similarity metric")
    ax2.spines['bottom'].set_visible(True)
    fig.suptitle("Aquired (Normalised) similarities, for registering images under different circumstances, \nusing CC and MI as a similarity metric. ")
    plt.subplots_adjust(bottom=0.15)
    fig.text(0.5, 0.005, """This figure shows the Normalised Similarity (S) after registration of pairs of images(that have different modalities: T1 and T2), 
             unders several sircumstances. This is shown for the usage of two similarity metrices: Mutual Information and Cross correlation. 
             The similarity is normalized by deviding S by the similarity of the two original images. This means that image pairs with an S above 1 are better aligned 
             then the original pair."""
             , ha='center', fontsize=11)


    plt.show()
    



df_normalized_MI = normalized_data('MI_data')
df_normalized_CC = normalized_data('CC_data')

plot_results(df_normalized_CC,df_normalized_MI)
