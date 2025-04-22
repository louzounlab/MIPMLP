from .create_otu_and_mapping_files import CreateOtuAndMappingFiles
import plotly.express as px


"""
This module defines the main preprocessing function used in the pipeline.
It initializes the data processing class, sets parameters, and performs optional plotting of bacteria presence.
"""

# Import data handling and plotting utilities
def preprocess(df, tag=None, taxonomy_level=7, taxnomy_group='mean', epsilon=0.1, normalization='log', z_scoring='No', norm_after_rel='No', pca=(0, 'PCA'),
               rare_bacteria_threshold=0.01, plot=False, df_test=None, external_sub_pca=None, external_pca=None,drop_tax_prefix=True):
    """
    Preprocess OTU and mapping data for downstream analysis.

    Parameters:
    - df: DataFrame of OTU table
    - tag: optional mapping table
    - taxonomy_level (int): level of taxonomy to use (4 to 7)
    - taxnomy_group (str): how to group taxonomy ('sub PCA', 'mean', 'sum')
    - epsilon (float): small value added to avoid log(0)
    - normalization (str): type of normalization ('log', 'relative')
    - z_scoring (str): z-score normalization ('row', 'col', 'both', 'No')
    - norm_after_rel (str): normalization after relative abundance ('No', 'relative')
    - pca (tuple): (0/1, 'PCA') to apply PCA
    - rare_bacteria_threshold (float): threshold to filter rare bacteria
    - plot (bool): whether to plot bacteria presence
    - df_test: optional test DataFrame
    - external_sub_pca: pre-fitted SubPCA object (optional)
    - external_pca: pre-fitted PCA object (optional)

    Returns:
    - Processed OTU features DataFrame (train)
    - Processed OTU features DataFrame (test, if provided)
    """

    # Pack all preprocessing parameters into a dictionary
    params = {
        'taxonomy_level': taxonomy_level,
        'taxnomy_group': taxnomy_group,
        'epsilon': epsilon,
        'normalization': normalization,
        'z_scoring': z_scoring,
        'norm_after_rel': norm_after_rel,
        'pca': pca,
        'rare_bacteria_threshold': rare_bacteria_threshold,
        'external_sub_pca': external_sub_pca,
        'external_pca': external_pca,
        'drop_tax_prefix': drop_tax_prefix
    }

    # Initialize the OTU and mapping file processor
    if tag is None:
        mapping_file = CreateOtuAndMappingFiles(df, tags_file=None, df_test=df_test)
    else:
        mapping_file = CreateOtuAndMappingFiles(df,tag, df_test=df_test)

    # Apply preprocessing steps to the data
    mapping_file.apply_preprocess(preprocess_params=params)

    if plot:       # Optionally generate a plot of bacteria presence across samples
        plot_bacteria_presence(mapping_file.otu_features_df, output_path="figures/bacteria_presence.png")

    if df_test is not None:
        return mapping_file.otu_features_df, mapping_file.otu_features_test_df
    else:
        return mapping_file.otu_features_df


# Function creates plot bar chart showing the percentage of samples each bacterium appears in
def plot_bacteria_presence(otu_df, output_path):
    # Remove taxonomy row if exists before calculating presence
    otu_df = otu_df.drop(index='taxonomy', errors='ignore')
    # Calculate the percentage of samples in which each bacterium is present
    presence_percent = (otu_df > 0).sum(axis=0) / otu_df.shape[0] * 100
    df = presence_percent.reset_index()
    df.columns = ['Bacteria', 'Percentage']
    df = df.sort_values(by='Percentage', ascending=False)

    # Create interactive bar chart using Plotly
    fig = px.bar(df,
                 x='Bacteria',
                 y='Percentage',
                 hover_data={'Bacteria': True, 'Percentage': True},
                 labels={'Percentage': 'Percentage of Samples (%)'},
                 title='Bacteria Presence Across Samples',
                 text=None)

    fig.update_layout(xaxis={'showticklabels': False}, height=600)
    fig.write_html(output_path) # Save the plot
    fig.show() # Show the plot




