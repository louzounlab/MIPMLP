import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# mapping the names of the group to indexes
def groups_to_dict(list):
    dict = {}
    i = 1
    for l in list:
        if l not in dict:
            dict[l] = i
            i+=1
    return dict

# plotting a boxplot by a given diversity file and a grouping file with either 1 or 2 columns
def plot_boxplot(diversity_csv_file,group_csv_file, diversity_name= "alpha diversity"):
    series_array = []
    groups = pd.read_csv(group_csv_file)
    diversities= pd.read_csv(diversity_csv_file)
    if len(groups.columns) != 2:
        print("bad format of diversity file")
        return None
    if len(groups.columns) > 2 or len(groups.columns) < 0:
        print("bad format of grouping file")
        return  None
    if len(groups.index) !=  len(diversities.index):
        print("length of groups file isnt matching the length of the the diversity file")
        return None
    if len(groups.columns) == 1:
        dictonary = groups_to_dict(list(groups.iloc[:, 0]))
        groups = groups.replace({0: dictonary})
        max_group = groups.max().max()
        for i in range(1,max_group+1):
            places = groups.iloc[:,0]== i
            series_array.append(list(diversities[places].iloc[:,1]))
    elif len(groups.columns) == 2:
        dictonary = groups_to_dict(list(groups.iloc[:, 1]))
        groups.iloc[:,1] = groups.iloc[:,1].replace(dictonary)
        max_group = groups.iloc[:, 1].max().max()
        for i in range(1, max_group + 1):
            l = list(groups.iloc[:,1] == i)
            series_array.append(list(diversities[l].iloc[:,1]))
    fig1, ax1 = plt.subplots()
    plt.ylabel(diversity_name)
    new_dict = {value: key for (key, value) in dictonary.items()}
    ax1.boxplot(series_array, labels=[new_dict[i] for i in range(1, groups.iloc[:, 1].max().max()+1)],  patch_artist=True, boxprops=dict(facecolor="red", color="blue"))
    plt.grid()
    return plt