


######## DEPRECATED: replaced by SubPCAByTaxonomy ##########


from sklearn.decomposition import PCA
import pandas as pd
from .general import apply_pca

"""
Applies per-group PCA dimensionality reduction based on taxonomy levels.

Parameters:
- perform_distance: whether to run distance learning
- level: taxonomy level to group features (e.g. genus, species)
- preprocessed_data: OTU table with samples as rows and taxa as columns
- mapping_file: metadata, passed through unchanged
- test_data: optional test OTU table to transform with PCA

Returns:
- new_df: dataframe after PCA per taxonomy group (train)
- mapping_file: unchanged
- new_df_test: dataframe after PCA (test), or None if not provided
"""

def distance_learning(perform_distance, level, preprocessed_data, mapping_file, test_data=None):

    new_df_test = pd.DataFrame(index=test_data.index) if test_data is not None else None

    if perform_distance:
        # Identify columns with a single unique value
        unique_cols = []
        # Keep constant-value columns to merge back later
        for c in range(preprocessed_data.shape[1]):
            m =  preprocessed_data.iloc[:,c].nunique()
            if preprocessed_data.iloc[:,c].nunique() == 1:
                unique_cols.append(preprocessed_data.columns[c])
        unique_cols_df = preprocessed_data[unique_cols]
        # Collect variable columns for PCA processing
        cols = []
        for c in range(preprocessed_data.shape[1]):
            m = preprocessed_data.iloc[:, c].nunique()
            if preprocessed_data.iloc[:, c].nunique() != 1:
                cols.append(preprocessed_data.columns[c])
        # Create dictionary to group features by taxonomy level
        dict_bact = {'else': []}
        for col in preprocessed_data[cols]:
            # Split taxonomy string into hierarchy levels
            col_name = col.split(';')
            # Skip placeholder or malformed taxonomy names
            bact_level = level - 1
            if col_name[0][-1] == '_':
                continue
            # Add column to the appropriate taxonomy group
            if len(col_name) > bact_level:
                while col_name[bact_level][-1] == "_":
                    bact_level-=1
                if ';'.join(col_name[:bact_level+1]) in dict_bact:
                    dict_bact[';'.join(col_name[:bact_level+1])].append(col)
                else:
                    dict_bact[';'.join(col_name[:bact_level+1])] = [col]
            else:
                dict_bact['else'].append(preprocessed_data[col].name)

        # Create new dataframe to hold PCA-reduced features
        new_df = pd.DataFrame(index=preprocessed_data.index)
        col = 0
        for key, values in dict_bact.items():
            if values:
                # Extract group of features for this taxonomy group
                new_data = preprocessed_data[values]
                # Initialize PCA with dynamic number of components
                pca = PCA(n_components=min(round(new_data.shape[1] / 2) + 1, new_data.shape[0]))
                pca.fit(new_data)
                sum = 0
                num_comp = 0
                # Determine number of components that explain >50% variance
                for (i, component) in enumerate(pca.explained_variance_ratio_):
                    if sum <= 0.5:
                        sum += component
                    else:
                        num_comp = i
                        break
                if num_comp == 0:
                    num_comp += 1
                # Apply PCA to the group
                otu_after_pca_new, pca_obj, pca_str = apply_pca(new_data, n_components=num_comp)

                if test_data is not None:
                    test_group = test_data[values]
                    test_after_pca = pd.DataFrame(pca.transform(test_group), index=test_data.index)


                # Add PCA components to the new dataframe with informative column names
                for j in range(otu_after_pca_new.shape[1]):
                    col_name = 'else;' if key == 'else' else str(
                        values[0][0:values[0].find(key) + len(key)]) + '_' + str(j)
                    new_df[col_name] = otu_after_pca_new[j]
                    if new_df_test is not None:
                        new_df_test[col_name] = test_after_pca.iloc[:, j]
                col += num_comp

        # Combine reduced PCA features with constant-value features
        dfs = [new_df, unique_cols_df]
        new_df = pd.concat(dfs, axis=1)

        # Return processed dataframe and unchanged mapping file
        return new_df, mapping_file, new_df_test
    else:
        return preprocessed_data, mapping_file, None

