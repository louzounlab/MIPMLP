import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import preprocess_grid

def plot_rel_freq(data_frame, taxonomy_col="taxonomy", tax_level=3, folder=None):
        taxonomy_reduced = data_frame[taxonomy_col].map(lambda x: x.split(';'))
        taxonomy_reduced = taxonomy_reduced.map(lambda x: ';'.join(x[:tax_level]))
        data_frame[taxonomy_col] = taxonomy_reduced
        data_frame = data_frame.groupby(data_frame[taxonomy_col]).mean()
        data_frame = data_frame.T
        data_frame = preprocess_grid.row_normalization(data_frame)
        plotting_with_pd(data_frame, folder, tax_level)


def plotting_with_pd(df: pd.DataFrame, folder=None, taxonomy_level=3):
    df = easy_otu_name(df)
    df = df.reindex(df.mean().sort_values().index, axis=1)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [3, 1]})
    ax2.axis('off')
    df.plot.bar(stacked=True, ax=ax, width=1.0, colormap='Spectral')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="x-small")
    ax.xaxis.set_ticks([])
    ax.set_xlabel("")
    ax.set_title("Relative frequency with taxonomy level "+str(taxonomy_level))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    #plt.show()
    plt.savefig(f"{folder}/relative_frequency_stacked.png")

def easy_otu_name(df):
    columns = []
    for i in df.columns:
        list = []
        for h in i.split(";"):
            h=h.lstrip(" kpcofgs")
            h=h.lstrip(" _")
            h=h.capitalize()
            list.append(h)
        name = str(list[-2:]).replace("[", "").replace("]", "").replace("\'", "")
        columns.append(name)
    df.columns = columns
    return df
