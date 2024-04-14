import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation


# mapping the names of the group to indexes
def groups_to_dict(list):
    dict = {}
    i = 1
    for l in list:
        if l not in dict:
            dict[l] = i
            i += 1
    return dict


# plotting a boxplot by a given diversity file and a grouping file with either 1 or 2 columns
def plot_boxplot(diversity, group_csv_file, diversity_name="alpha diversity"):
    series_array = []
    groups = group_csv_file
    g0 = groups[groups["Tag"] == 0.0]
    g1 = groups[groups["Tag"] == 1.0]
    d0 = diversity.loc[g0.index]
    d1 = diversity.loc[g1.index]
    boxes = [list(d0.values), list(d1.values)]
    plt.figure(1, figsize=(3, 3))
    ax = sns.swarmplot(data=boxes,
                       palette=["blue", "red"],
                       zorder=0)
    ax = sns.boxplot(data=boxes, boxprops={'facecolor': 'None'})
    test_results = add_stat_annotation(ax, data=pd.DataFrame(data=boxes, index=["A", "B"]).T,
                                       box_pairs=[("A", "B")],
                                       perform_stat_test=True,
                                       test="t-test_ind", text_format='star',
                                       loc='inside', verbose=2)
    plt.xticks(ticks=np.arange(2),
               labels=["Control", f"Condition"], fontsize=15, family='Times New Roman')
    plt.yticks(fontsize=15, family='Times New Roman')
    plt.ylabel(diversity_name, fontsize=15, family='Times New Roman')
    plt.show()


def adjustments(data: pd.DataFrame):
    """Adjust mipMLP's output to a state which we can input it to calculate_hill_numbers from."""
    data = data.T
    data.index = data["taxonomy"]
    del data["taxonomy"]
    data = data.astype(float)
    data = data.groupby(data.index).sum()
    data = data.T
    return data


def calculate_hill_numbers(df, q):
    """
    Calculate Hill numbers for each sample in a DataFrame.

    Parameters:
    - df: pandas.DataFrame
        The DataFrame containing abundance data (samples as rows, species as columns).
    - q: float
        The order of the Hill number.

    Returns:
    - hill_series: pandas.Series
        A Series containing the Hill number for each sample.
    """
    hill_values = []

    for sample in df.index:
        row_values = df.loc[sample].values
        row_values = np.array(row_values, dtype=float)

        if q != 0 and q != 1:
            # General Hill number calculation (Rényi entropy)
            hill_value = 1 / (1 - q) * np.log(np.sum(row_values ** q))
        elif q == 0:
            # Handle q=0 (species richness)
            hill_value = np.sum(row_values > 0)
        elif q == 1:
            # Normalize the values to probabilities
            prob_values = row_values / np.sum(row_values)

            # Calculate Shannon entropy
            hill_value = -np.sum(prob_values * np.log(prob_values + 1e-10))

        hill_values.append(hill_value)

    hill_series = pd.Series(hill_values, index=df.index)
    return hill_series


def plot_diversity(df, tag, q):
    df = adjustments(df)
    hill_series = calculate_hill_numbers(df, q)
    plot_boxplot(hill_series, tag, diversity_name=f"Hill{q}")
    return


if __name__ == "__main__":
    data = pd.read_csv("data/ibd_for_process.csv", index_col=0)
    tag = pd.read_csv("data/ibd_tag.csv", index_col=0)
    q = 1
    plot_diversity(data, tag, q)
