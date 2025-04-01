from .preprocess_grid import preprocess_data

"""
This module defines the CreateOtuAndMappingFiles class, which is responsible for handling microbiome data
preprocessing. It accepts an OTU table and an optional mapping (tags) file, and performs preprocessing steps
including:

- Taxonomy filtering and formatting
- Normalization (log or relative)
- Z-score standardization
- Optional PCA (dimensionality reduction)
- Correlation filtering
- Tag alignment and metadata processing

The processed OTU table is stored internally and can be used for downstream analysis or visualization.
"""


class CreateOtuAndMappingFiles(object):   # Class to manage OTU and mapping data and apply preprocessing
    def __init__(self, otu_file, tags_file):  # Get two relative path of csv files
        self.tags = False

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
        self.otu_features_df = otu_file.drop('Unnamed: 0', axis=1,errors='ignore')
        self.otu_features_df = self.otu_features_df.set_index('ID')
        self.otu_features_df.index = self.otu_features_df.index.astype(str)
        self.pca_ocj = None
        self.pca_comp = None

    def apply_preprocess(self, preprocess_params):    # Apply preprocessing steps including normalization and optional PCA
        if self.tags:
            self.otu_features_df, self.otu_features_df_b_pca, self.pca_ocj, self.bacteria, self.pca_comp = preprocess_data(
                self.otu_features_df, preprocess_params, self.tags_df)
        else:
            self.otu_features_df, self.otu_features_df_b_pca, self.pca_ocj, self.bacteria, self.pca_comp = preprocess_data(
                self.otu_features_df, preprocess_params, map_file=None)
                # otu_features_df is the processed data, before pca
        if int(preprocess_params['pca'][0]) == 0:
            self.otu_features_df = self.otu_features_df_b_pca
