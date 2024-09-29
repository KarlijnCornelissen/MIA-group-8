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
            df.loc[i]=(df.loc[i])-(df.loc[i,"Original images"])
    # s=[f"image: {i}" for i in range(1,7)]
    # df=df.loc[s]

    normalized_title = f"{title}_normalized"
    file_path = f"MIA-group-8/code/{normalized_title}.csv"

    # Save the DataFrame to a CSV file
    df.to_csv(file_path)
    return df




def make_boxplots(df):
    sns.set_theme(style="ticks", palette="pastel")
    print(pd.melt(df))

    
    # Draw a nested boxplot to show bills by day and time
    sns.catplot(x="variable", y="value",
                data=pd.melt(df), kind="violin", hue="variable")
    plt.show()

df_normalized = normalized_data('CC_data')
make_boxplots(df_normalized)

