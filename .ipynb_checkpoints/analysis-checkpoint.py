#Data analysis 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df_CC=pd.read_csv("MIA-group-8\code\CC_data.csv", index_col=[0,1])


# Creating normalized version:
def normalized_data(title):
    df=pd.read_csv(f"MIA-group-8\code\{title}.csv", index_col=[0,1])
    df.reset_index(drop=True)

    index_names= [f"image: {i}" for i in range(1,len(df.iloc[:,0])+1)]
    df["Image number"]=index_names
    
    df.set_index("Image number",inplace=True)
    df.columns = ["Original images", "high noise", "Low noise", "large filter", "small filter"]
    for i in index_names:
        for j in df.columns:
            df.loc[i]=(df.loc[i])/(df.loc[i,"Original images"])

    # if True: #title == "MI":
    #     s=[f"image: {i}" for i in range(1,7)]
    #     df=df.loc[s]


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

def make_boxplots(df_CC, df_MI):
    sns.set_theme(style="ticks", palette="pastel")
    

    
    # Draw a nested boxplot to show bills by day and time
    # sns.catplot(x="variable", y="value",
    #             data=pd.melt(df), kind="violin", hue="variable")
    # plt.show()

    fig = plt.figure(figsize=(6, 6))
    ax1= plt.subplot(211)
    
    
    
    #CC:
    df_CC = pd.melt(df_CC)
    df_CC.columns = ["Image type", "Normalized Similarity" ]

    sns.stripplot(ax=ax1, 
        data=df_CC, x="Image type", y="Normalized Similarity", hue="Image type",
        )
    sns.pointplot(ax=ax1,
        data=df_CC, x="Image type", y="Normalized Similarity", hue="Image type",
        dodge=.8 - .8 / 3, palette="dark", errorbar=None,
        markers="d", markersize=4, linestyle="none",
        )
    # MI:
    ax2 = plt.subplot(212)
    df_MI = pd.melt(df_MI)
    df_MI.columns = ["Image type", "Normalized Similarity" ]

    sns.stripplot(ax=ax2,
        data=df_MI, x="Image type", y="Normalized Similarity", hue="Image type",
        )
    sns.pointplot(ax=ax2,
        data=df_MI, x="Image type", y="Normalized Similarity", hue="Image type",
        dodge=.8 - .8 / 3, palette="dark", errorbar=None,
        markers="d", markersize=4, linestyle="none",
        ) 
    
    sns.despine(left=True)
    plt.show()
    
def make_heatmap(df):
    sns.heatmap(df, annot=True)
    plt.show()



df_normalized_MI = normalized_data('MI_data')
df_normalized_CC = normalized_data('CC_data')
# 
make_boxplots(df_normalized_CC,df_normalized_MI)
#print(df_normalized.columns[0:5].values())
#make_scatterplot(df_normalized)
make_heatmap(df_normalized)
