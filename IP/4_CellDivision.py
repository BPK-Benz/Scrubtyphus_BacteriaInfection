import os,glob
import pandas as pd
import numpy as np



# Cell division
# 1. Eccentricity (Ellipticity) > 0.8 (ranging between zero to one)
# 2. Intensity > mean+std (per image)

def divideCell(df):
    temp_divide =  df
    threshold = temp_divide['Nucleus_Intensity'].mean()+temp_divide['Nucleus_Intensity'].std()
    temp_divide['Intensity_Threshold'] = temp_divide['Nucleus_Intensity']>threshold
    keep = []
    for _ in range(temp_divide.shape[0]):
        if temp_divide.loc[_,'Nucleus_Eccentricity'] > 0.8:
            if temp_divide.loc[_,'Intensity_Threshold'] == True:
                keep = keep + [True]
            else:
                keep = keep + [False]
        else:
            keep = keep + [False]
    temp_divide['divide'] = keep
    return temp_divide


whole = glob.glob(os.getcwd()+'/P*/T*/*')+ glob.glob(os.getcwd()+'/P*/Scram*/*')
infectList = []
for _ in range(len(whole)):
    temp_divide = pd.DataFrame()
    temp  =  pd.read_csv(whole[_])
    count = divideCell(df=temp).iloc[:,-4:-1]
    count_border = 0
    count_uninfect = 0
    count_infect = 0
    count_divide = 0

    for x in range(count.shape[0]):
        if count.iloc[x,0] == 1:
             count_border+=1
        elif count.iloc[x,1] == True:
            count_divide+=1
        else:
            if 'non' in count.iloc[x,-1]:
                count_uninfect+=1

            else:
                count_infect+=1
                
    infectList = infectList + [[whole[_].split('/S1/')[1][:-11],count_border, count_divide,count_uninfect,count_infect]]

path = 'path_output'
pd.DataFrame(infectList, columns=['Gene','Border','Divide','Uninfect','Infect']).to_csv(path+'IP_InfectedCell.csv')