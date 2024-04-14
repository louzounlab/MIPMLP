# created by Yoel Jasner
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import Counter
from .distance_learning_func import distance_learning
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
taxonomy_col = 'taxonomy'
min_letter_value = 'a'

states = {1: "Creating otu And Mapping Files",
          2:"Perform taxonomy grouping",
3:'Perform normalization',
4:'dimension reduction',
5:"plotting diversities"}

def update_state(ip, position):
    with open("./" + str(ip) + "/state.txt", "w+") as f:
        f.write(str(position) + "\n" + states[position])

def preprocess_data(data, dict_params: dict, map_file):
    taxonomy_level = int(dict_params['taxonomy_level'])
    preform_taxnomy_group = dict_params['taxnomy_group']
    eps_for_zeros = float(dict_params['epsilon'])
    preform_norm = dict_params['normalization']
    preform_z_scoring = dict_params['z_scoring']
    relative_z = dict_params['norm_after_rel']
    correlation_removal_threshold = dict_params.get('correlation_threshold', None)
    rare_bacteria_threshold = dict_params.get('rare_bacteria_threshold', None)
    pca = dict_params['pca']



    as_data_frame = pd.DataFrame(data.T).apply(pd.to_numeric, errors='ignore').copy()  # data frame of OTUs
    as_data_frame = as_data_frame.fillna(0)
    as_data_frame.columns = [';'.join(str(i).split(';')[:7]) for i in as_data_frame.columns ]

    #handling edge cases - droping viruese, unclastered bacterias, bacterias which are clustered with more than specie and unnamed bacterias
    indexes = as_data_frame[taxonomy_col]
    stay = []
    for i in range(len(indexes)):
        if str(as_data_frame[taxonomy_col][i])[0].lower() > min_letter_value and as_data_frame[taxonomy_col][i].split(';')[0][-len("Viruses"):] != "Viruses":
            length = len(as_data_frame[taxonomy_col][i].split(';'))
            if length<8 and not ("." not in as_data_frame[taxonomy_col][i].split(';')[length-1] and as_data_frame[taxonomy_col][i].split(';')[length-1][-1]!="_" and check_cluster(as_data_frame[taxonomy_col][i].split(';'))):
                stay.append(i)

    as_data_frame = as_data_frame.iloc[stay,:]

    # filling empty taxonomy levels
    as_data_frame = fill_taxonomy(as_data_frame, tax_col=taxonomy_col)

    # droping space from the taxonomy name
    indexes = as_data_frame[taxonomy_col]
    new_indexes = []
    for i in range(len(indexes)):
        new_indexes.append(as_data_frame[taxonomy_col][i].replace(" ",""))
    as_data_frame[taxonomy_col] = new_indexes
    as_data_frame.index = new_indexes

    # updating state - Performing taxonomy grouping

    if preform_taxnomy_group != '':
        as_data_frame = taxonomy_grouping(as_data_frame, preform_taxnomy_group, taxonomy_level)

        # here the samples are columns
        as_data_frame = as_data_frame.T
    else:
        try:
            as_data_frame = as_data_frame.drop(taxonomy_col, axis=1).T
            # here the samples are columns
        except:
            pass

    # remove highly correlated bacteria
    if correlation_removal_threshold is not None:
        as_data_frame = dropHighCorr(as_data_frame, correlation_removal_threshold)

    # drop bacterias with single values
    if rare_bacteria_threshold is not None:
        as_data_frame = drop_rare_bacteria(as_data_frame, rare_bacteria_threshold)

    if preform_norm == 'log':
        as_data_frame = log_normalization(as_data_frame, eps_for_zeros)


        if preform_z_scoring != 'No':
            as_data_frame = z_score(as_data_frame, preform_z_scoring)


    elif preform_norm == 'relative':
        as_data_frame = row_normalization(as_data_frame)
        if relative_z == "z_after_relative":
            as_data_frame = z_score(as_data_frame, 'col')

    as_data_frame_b_pca = as_data_frame.copy()
    bacteria = as_data_frame.columns
    if preform_taxnomy_group == 'sub PCA':
        # as_data_frame.columns = taxo_col
        as_data_frame, _ = distance_learning(perform_distance=True, level=taxonomy_level,
                                             preproccessed_data=as_data_frame, mapping_file=map_file)
        as_data_frame_b_pca = as_data_frame
        as_data_frame = fill_taxonomy(as_data_frame, tax_col='columns')

    # as_data_frame.columns = [delete_empty_taxonomic_levels(i) for i in as_data_frame.columns]
    #as_data_frame_b_pca.columns = [delete_empty_taxonomic_levels(i) for i in as_data_frame_b_pca.columns]


    # updating state - performing pca
    if pca[0] != 0:
        as_data_frame, pca_obj, pca = apply_pca(as_data_frame, n_components=pca[0], dim_red_type=pca[1])
    else:
        pca_obj = None

    return as_data_frame, as_data_frame_b_pca, pca_obj, bacteria, pca


def row_normalization(as_data_frame):
    as_data_frame = as_data_frame.div(as_data_frame.sum(axis=1), axis=0).fillna(0)
    return as_data_frame


def drop_low_var(as_data_frame, threshold):
    drop_list = [col for col in as_data_frame.columns if col != 'taxonomy' and threshold > np.var(as_data_frame[col])]
    return as_data_frame.drop(columns=drop_list).T


def log_normalization(as_data_frame, eps_for_zeros):
    as_data_frame = as_data_frame.astype(float)
    as_data_frame += eps_for_zeros
    as_data_frame = np.log10(as_data_frame)
    return as_data_frame


def z_score(as_data_frame, preform_z_scoring):
    if preform_z_scoring == 'row':
        # z-score on columns
        as_data_frame[:] = preprocessing.scale(as_data_frame, axis=1)
    elif preform_z_scoring == 'col':
        # z-score on rows
        as_data_frame[:] = preprocessing.scale(as_data_frame, axis=0)
    elif preform_z_scoring == 'both':
        as_data_frame[:] = preprocessing.scale(as_data_frame, axis=1)
        as_data_frame[:] = preprocessing.scale(as_data_frame, axis=0)

    return as_data_frame


def drop_bacteria(as_data_frame):
    bacterias = as_data_frame.columns
    bacterias_to_dump = []
    for i, bact in enumerate(bacterias):
        f = as_data_frame[bact]
        num_of_different_values = set(f)
        if len(num_of_different_values) < 2:
            bacterias_to_dump.append(bact)
    if len(bacterias_to_dump) != 0:
        print("number of bacterias to dump before intersection: " + str(len(bacterias_to_dump)))
        print("percent of bacterias to dump before intersection: " + str(
            len(bacterias_to_dump) / len(bacterias) * 100) + "%")
    else:
        print("No bacteria with single value")
    return as_data_frame.drop(columns=bacterias_to_dump)


def dropHighCorr(data, threshold):
    corr = data.corr()
    df_not_correlated = ~(corr.mask(np.tril(np.ones([len(corr)] * 2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = data[un_corr_idx]
    number_of_bacteria_dropped = len(data.columns) - len(df_out.columns)
    return df_out


def drop_rare_bacteria(as_data_frame, threshold):
    bact_to_num_of_non_zeros_values_map = {}
    bacteria = as_data_frame.columns
    num_of_samples = len(as_data_frame.index) - 1
    for bact in bacteria:
        values = as_data_frame[bact]
        count_map = Counter(values)
        zeros = 0
        if 0 in count_map.keys():
            zeros += count_map[0]
        if '0' in count_map.keys():
            zeros += count_map['0']

        bact_to_num_of_non_zeros_values_map[bact] = num_of_samples - zeros

    rare_bacteria = []
    for key, val in bact_to_num_of_non_zeros_values_map.items():
        if val < threshold:
            rare_bacteria.append(key)
    as_data_frame.drop(columns=rare_bacteria, inplace=True)
    return as_data_frame


def apply_pca(data, n_components=15, dim_red_type='PCA', visualize=False):
    if n_components == -1:
        pca = PCA(n_components=min(len(data.index), len(data.columns)))
        pca.fit(data)
        data_components = pca.fit_transform(data)
        for accu_var, (i, component) in zip(pca.explained_variance_ratio_.cumsum(),
                                            enumerate(pca.explained_variance_ratio_)):
            if accu_var > 0.7:
                components = i + 1
                break
    else:
        components = n_components
    if dim_red_type == 'PCA':
        pca = PCA(n_components=components)
        pca.fit(data)
        data_components = pca.fit_transform(data)

        str_to_print = str("Explained variance per component: \n" +
                           '\n'.join(['Component ' + str(i) + ': ' +
                                      str(component) + ', Accumalative variance: ' + str(accu_var) for
                                      accu_var, (i, component) in zip(pca.explained_variance_ratio_.cumsum(),
                                                                      enumerate(pca.explained_variance_ratio_))]))

        str_to_print += str("\nTotal explained variance: " + str(pca.explained_variance_ratio_.sum()))

        if visualize:
            plt.figure()
            plt.plot(pca.explained_variance_ratio_.cumsum())
            plt.bar(np.arange(0, components), height=pca.explained_variance_ratio_)
            plt.title(
                f'PCA - Explained variance using {n_components} components: {pca.explained_variance_ratio_.sum()}')
            plt.xlabel('PCA #')
            plt.xticks(list(range(0, components)), list(range(1, components + 1)))

            plt.ylabel('Explained Variance')
    else:
        pca = FastICA(n_components=components)
        data_components = pca.fit_transform(data)
    return pd.DataFrame(data_components).set_index(data.index), pca, components


def fill_taxonomy(as_data_frame, tax_col):
    if tax_col == 'columns':
        df_tax = pd.Series(as_data_frame.columns).str.split(';', expand=True)
        i = df_tax.shape[1]
        while i < 7:
            df_tax[i] = np.nan
            i+=1
    else:
        df_tax = as_data_frame[tax_col].str.split(';', expand=True)
        if df_tax.shape[1] == 1:
            # We need to use a differant separator
            df_tax = as_data_frame[tax_col].str.split('|', expand=True)
    df_tax[6] = df_tax[6].fillna('s__')
    df_tax[5] = df_tax[5].fillna('g__')
    df_tax[4] = df_tax[4].fillna('f__')
    df_tax[3] = df_tax[3].fillna('o__')
    df_tax[2] = df_tax[2].fillna('c__')
    df_tax[1] = df_tax[1].fillna('p__')
    df_tax[0] = df_tax[0].fillna('s__')
    if tax_col == 'columns':
        as_data_frame.columns = df_tax[0] + ';' + df_tax[1] + ';' + df_tax[2
        ] + ';' + df_tax[3] + ';' + df_tax[4] + ';' + df_tax[5] + ';' + df_tax[
                                     6]
    else:
        as_data_frame[tax_col] = df_tax[0] + ';' + df_tax[1] + ';' + df_tax[2
        ] + ';' + df_tax[3] + ';' + df_tax[4] + ';' + df_tax[5] + ';' + df_tax[
                                 6]
    return as_data_frame




def from_biom(biom_file_path, taxonomy_file_path, otu_dest_path, **kwargs):
    # Load the biom table and rename index.
    from biom import load_table
    otu_table = load_table(biom_file_path).to_dataframe(True)
    # Load the taxonomy file and extract the taxonomy column.
    taxonomy = pd.read_csv(taxonomy_file_path, index_col=0, sep=None, **kwargs).drop('Confidence', axis=1,
                                                                                     errors='ignore')
    otu_table = pd.merge(otu_table, taxonomy, right_index=True, left_index=True)
    otu_table.rename({'Taxon': 'taxonomy'}, inplace=True, axis=1)
    otu_table = otu_table.transpose()
    otu_table.index.name = 'ID'
    otu_table.to_csv(otu_dest_path)

def taxonomy_grouping(as_data_frame, preform_taxnomy_group, taxonomy_level):
    taxonomy_reduced = as_data_frame[taxonomy_col].map(lambda x: x.split(';'))
    if preform_taxnomy_group == 'sub PCA':
        taxonomy_reduced = taxonomy_reduced.map(lambda x: ';'.join(x[:]))
    else:
        taxonomy_reduced = taxonomy_reduced.map(lambda x: ';'.join(x[:taxonomy_level]))
    as_data_frame[taxonomy_col] = taxonomy_reduced
    # group by mean
    if preform_taxnomy_group == 'mean':
        as_data_frame = as_data_frame.groupby(as_data_frame[taxonomy_col]).mean()
    # group by sum
    elif preform_taxnomy_group == 'sum':
        as_data_frame = as_data_frame.groupby(as_data_frame[taxonomy_col]).sum()
        # group by anna PCA
    elif preform_taxnomy_group == 'sub PCA':
        taxo_col = as_data_frame['taxonomy']
        # as_data_frame = as_data_frame.iloc[:,:-1]
        as_data_frame = as_data_frame.groupby(as_data_frame[taxonomy_col]).mean()
    return as_data_frame

def delete_empty_taxonomic_levels(i):
    splited = i.split(';')
    while re.search(r'^[a-z]_+\d*$', splited[-1]) is not None:
        splited = splited[:-1]
    i = ""
    for j in splited:
        i += j
        i += ';'
    i = i[:-1]
    return i

def check_cluster(tax):
    length = len(tax)
    length = length- 2
    while length >= 0:
        if tax[length][-1] == '_':
            return True
        length-=1
    return False