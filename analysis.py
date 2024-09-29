#Data analysis 
import pandas as pd
import numpy as np
df_CC_test=pd.read_csv("MIA-group-8\code\CC_data_test.csv",index_col=[0,1])
print(df_CC_test)

# Creating normalized version:
def normalized_data(title):
    df=pd.read_csv(f"{title}.csv")
    df_normalized = pd.DataFrame(np.nan, index=df.index, columns=df.columns)

    for i in df.index:
        df_normalized.iloc[i, 0] = df.iloc[i, 0]
        for j in range(1,df.columns+1):
            df_normalized.iloc[i,j]=(df.iloc[i,j])/(df.iloc[i,0])



    normalized_title = f"{title}_normalized"
    file_path = f"MIA-group-8/code/{normalized_title}.csv"

    # Save the DataFrame to a CSV file
    df_normalized.to_csv(file_path)


normalized_data('CC_data_test')
