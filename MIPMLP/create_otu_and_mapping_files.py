from .preprocess_grid import preprocess_data

"""
CreateOtuAndMappingFiles handles preprocessing of microbiome OTU data 
for machine learning. Supports both train and optional test sets.

Main usage:
- Initialize with train OTU table (+ optional tags + optional test OTU table)
- Run apply_preprocess() to perform all transformations on train,
  and optionally transform test with the same fitted steps.

__init__(df_train, tags_file=None, df_test=None)
    - Loads and formats the OTU train data
    - Optionally processes tags file (for labels/metadata)
    - Optionally loads test OTU data (will only be transformed later)

apply_preprocess(preprocess_params)
    - Applies preprocessing pipeline to train
    - If test data was provided, applies same transforms to test
    - Updates internal attributes with processed train and test data
"""


class CreateOtuAndMappingFiles(object):   # Class to manage OTU and mapping data and apply preprocessing
    def __init__(self, df_train, tags_file, df_test=None):  # Get two relative path of csv files
        self.tags = False
        self.otu_features_test_df = None
        self.external_sub_pca = None
        self.external_pca = None


        if tags_file is not None:    # Check if mapping (tags) file was provided
            self.tags = True
            mapping_table = tags_file
            self.extra_features_df = mapping_table.drop(['Tag'], axis=1).copy()  # Separate features from the tag column
            self.tags_df = mapping_table[['Tag']].copy()     # Set the index of tags to be the sample ID
            self.tags_df.index = self.tags_df.index.astype(str)
            # Collect all sample IDs from the mapping file
            self.ids = self.tags_df.index.tolist()
            self.ids.append('taxonomy')

        # Prepare OTU features DataFrame: remove unnamed column and set index
        # ======== train df ========
        self.otu_features_df = df_train.drop('Unnamed: 0', axis=1, errors='ignore')
        self.otu_features_df = self.otu_features_df.set_index('ID')
        self.otu_features_df.index = self.otu_features_df.index.astype(str)
        # ======== test df (if provided) ========
        if df_test is not None:
            self.otu_features_test_df = df_test.drop('Unnamed: 0', axis=1, errors='ignore')
            self.otu_features_test_df = self.otu_features_test_df.set_index('ID')
            self.otu_features_test_df.index = self.otu_features_test_df.index.astype(str)

        self.pca_ocj = None
        self.pca_comp = None
        self.sub_pca_ocj = None

    # Apply preprocessing steps including normalization and optional PCA
    def apply_preprocess(self, preprocess_params):
        self.external_sub_pca = preprocess_params.get("external_sub_pca", None)
        self.external_pca = preprocess_params.get("external_pca", None)

        if self.otu_features_test_df is not None:
            result = preprocess_data(
                                     self.otu_features_df,
                                     preprocess_params,
                                     self.tags_df if self.tags else None,
                                     data_test=self.otu_features_test_df
                                     )
        elif self.tags:
            result = preprocess_data(
                self.otu_features_df,
                preprocess_params,
                self.tags_df
            )
        else:
            result = preprocess_data(
                self.otu_features_df,
                preprocess_params,
                map_file=None
            )

        # Unpack results, with or without test
        if len(result) == 7:
            (self.otu_features_df,
             self.otu_features_df_b_pca,
             self.pca_ocj,
             self.bacteria,
             self.pca_comp,
             self.otu_features_test_df,
             self.sub_pca_ocj) = result

        elif len(result) == 6:
            (self.otu_features_df,
             self.otu_features_df_b_pca,
             self.pca_ocj,
             self.bacteria,
             self.pca_comp,
             self.sub_pca_ocj) = result
            self.otu_features_test_df = None

        else:
            (self.otu_features_df,
             self.otu_features_df_b_pca,
             self.pca_ocj,
             self.bacteria,
             self.pca_comp) = result
            self.otu_features_test_df = None
            self.sub_pca_ocj = None