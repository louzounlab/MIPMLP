import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from .subpca_by_taxonomy import SubPCAByTaxonomy


taxonomy_col = 'taxonomy'
min_letter_value = 'a'

states = {1: "Creating otu And Mapping Files",
          2:"Perform taxonomy grouping",
          3:'Perform normalization',
          4:'dimension reduction',
          5:"plotting diversities"}


def preprocess_data(data, dict_params: dict, map_file=None, data_test=None):
    """
    Main preprocessing function for microbiome OTU data.

    Parameters:
    - data: DataFrame of OTU counts (df_train if provided)
    - dict_params: dictionary of preprocessing options
    - map_file: mapping table with sample tags (can be None)

    Returns:
    - processed OTU DataFrame
    - OTU DataFrame before PCA
    - PCA object (if PCA applied)
    - list of bacteria (features) used
    - PCA configuration tuple
    """
    # Unpack preprocessing parameters from the user-defined dictionary
    taxonomy_level = int(dict_params['taxonomy_level'])
    preform_taxnomy_group = dict_params['taxnomy_group']
    eps_for_zeros = float(dict_params['epsilon'])
    preform_norm = dict_params['normalization']
    preform_z_scoring = dict_params['z_scoring']
    relative_z = dict_params['norm_after_rel']
    correlation_removal_threshold = dict_params.get('correlation_threshold', None)
    rare_bacteria_threshold = dict_params.get('rare_bacteria_threshold', None)
    pca = dict_params['pca']

    # Convert data to numeric DataFrame and limit taxonomy to 7 levels
    as_data_frame = pd.DataFrame(data.T).apply(pd.to_numeric, errors='ignore').copy()  # data frame of OTUs
    as_data_frame = as_data_frame.fillna(0)
    as_data_frame.columns = [';'.join(str(i).split(';')[:7]) for i in as_data_frame.columns ]

    # Filter out non-bacterial or poorly classified entries- droping viruese, unclasstered bacterias, bacterias which are clustered with more than specie and unnamed bacterias
    indexes = as_data_frame[taxonomy_col]
    stay = []
    for i in range(len(indexes)):
        if str(as_data_frame[taxonomy_col][i])[0].lower() > min_letter_value and as_data_frame[taxonomy_col][i].split(';')[0][-len("Viruses"):] != "Viruses":
            length = len(as_data_frame[taxonomy_col][i].split(';'))
            if length<8 and not ("." not in as_data_frame[taxonomy_col][i].split(';')[length-1] and
                                 as_data_frame[taxonomy_col][i].split(';')[length-1][-1]!="_" and
                                 check_cluster(as_data_frame[taxonomy_col][i].split(';'))):
                stay.append(i)

    as_data_frame = as_data_frame.iloc[stay,:]

    # Perform taxonomy grouping (mean, sum, or sub PCA) if requested
    if preform_taxnomy_group != '':
        as_data_frame = taxonomy_grouping(as_data_frame, preform_taxnomy_group, taxonomy_level)
        as_data_frame = as_data_frame.T

    # לClean taxonomy names and drop taxonomy column
    as_data_frame = clean_taxonomy_names(as_data_frame)

    # Remove features (bacteria) with high correlation
    if correlation_removal_threshold is not None:
        as_data_frame = dropHighCorr(as_data_frame, correlation_removal_threshold)

    # Remove bacteria that appear in too few samples
    if rare_bacteria_threshold is not None:
        as_data_frame = drop_rare_bacteria(as_data_frame, rare_bacteria_threshold)

    # Apply normalization: log or relative abundance
    if preform_norm == 'log':
        as_data_frame = log_normalization(as_data_frame, eps_for_zeros)
        # Optionally apply z-score normalization
        if preform_z_scoring != 'No':
            as_data_frame = z_score(as_data_frame, preform_z_scoring)
    elif preform_norm == 'relative':
        as_data_frame = row_normalization(as_data_frame)
        if relative_z == "z_after_relative":
            as_data_frame = z_score(as_data_frame, 'col')

    # Store a copy of the data before PCA for optional use
    as_data_frame_b_pca = as_data_frame.copy()
    bacteria = as_data_frame.columns

    # If using sub PCA, apply distance learning to generate features
    if preform_taxnomy_group == 'sub PCA':
        sub_pca = SubPCAByTaxonomy(level=taxonomy_level)
        as_data_frame = sub_pca.fit(as_data_frame)

        as_data_frame_b_pca = as_data_frame
        as_data_frame = fill_taxonomy(as_data_frame, tax_col='columns')

    # Apply PCA if requested (dimensionality reduction)
    if pca[0] != 0:
        as_data_frame, pca_obj, pca = apply_pca(as_data_frame, n_components=pca[0], dim_red_type=pca[1])
    else:
        pca_obj = None

    # If test data provided, apply transform using train parameters
    if data_test is not None:
        test_data_frame = pd.DataFrame(data_test.T).apply(pd.to_numeric, errors='ignore').copy()
        test_data_frame = test_data_frame.fillna(0)
        test_data_frame.columns = [';'.join(str(i).split(';')[:7]) for i in test_data_frame.columns]
        test_data_frame = clean_taxonomy_names(test_data_frame)
        test_data_frame = test_data_frame.T
        if preform_taxnomy_group == 'sub PCA':
            test_data_frame = sub_pca.transform(test_data_frame)

        # Apply same normalization steps
        if preform_norm == 'log':
            test_data_frame = log_normalization(test_data_frame, eps_for_zeros)
            if preform_z_scoring != 'No':
                test_data_frame = z_score(test_data_frame, preform_z_scoring)
        elif preform_norm == 'relative':
            test_data_frame = row_normalization(test_data_frame)
            if relative_z == "z_after_relative":
                test_data_frame = z_score(test_data_frame, 'col')

        # Apply PCA transform if PCA was trained
        if pca_obj is not None:
            test_data_frame = pca_obj.transform(test_data_frame)
            test_data_frame = pd.DataFrame(test_data_frame, index=data_test.index)
        return as_data_frame, as_data_frame_b_pca, pca_obj, bacteria, pca, test_data_frame
    return as_data_frame, as_data_frame_b_pca, pca_obj, bacteria, pca, None


# Normalize each sample's row to sum to 1 (relative abundance)
def row_normalization(as_data_frame):
    as_data_frame = as_data_frame.div(as_data_frame.sum(axis=1), axis=0).fillna(0)
    return as_data_frame


# Apply log10 normalization with epsilon to avoid log(0)
def log_normalization(as_data_frame, eps_for_zeros):
    as_data_frame = as_data_frame.astype(float)
    as_data_frame += eps_for_zeros
    as_data_frame = np.log10(as_data_frame)
    return as_data_frame


# Apply z-score normalization across rows, columns, or both
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


# Drop bacteria that are highly correlated with others beyond a threshold
def dropHighCorr(data, threshold):
    corr = data.corr()
    df_not_correlated = ~(corr.mask(np.tril(np.ones([len(corr)] * 2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = data[un_corr_idx]
    number_of_bacteria_dropped = len(data.columns) - len(df_out.columns)
    return df_out


# Remove bacteria that appear in fewer samples than the defined threshold
def drop_rare_bacteria(as_data_frame, threshold):
    threshold = threshold * len(as_data_frame)  #threshold as number of people not as percent
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


# Apply PCA or ICA to reduce dimensions and return transformed data
def apply_pca(data, n_components=15, dim_red_type='PCA'):
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

    else:
        pca = FastICA(n_components=components)
        data_components = pca.fit_transform(data)
    return pd.DataFrame(data_components).set_index(data.index), pca, components


# Fill missing taxonomy levels (e.g., genus, species) with placeholders
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


#  Clean taxonomy names (standardize + index)
def clean_taxonomy_names(df):
    if taxonomy_col not in df.columns:
        # Skip if taxonomy already dropped
        return df

    df = fill_taxonomy(df, tax_col=taxonomy_col)
    df.index = df[taxonomy_col].str.replace(" ", "")
    df = df.drop(taxonomy_col, axis=1)
    return df




# Group features based on taxonomy level and aggregation method
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
        as_data_frame = as_data_frame.groupby(as_data_frame[taxonomy_col]).mean()
    return as_data_frame



# Check if a taxonomy string has intermediate levels marked as '_'
def check_cluster(tax):
    length = len(tax)
    length = length- 2
    while length >= 0:
        if tax[length][-1] == '_':
            return True
        length-=1
    return False