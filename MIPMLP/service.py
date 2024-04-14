from .create_otu_and_mapping_files import CreateOtuAndMappingFiles
preprocess_prms = {'taxonomy_level': 6, 'taxnomy_group': 'sub PCA', 'epsilon': 0.1,
                   'normalization': 'log', 'z_scoring': 'row', 'norm_after_rel': 'No',
                   'std_to_delete': 0, 'pca': (0, 'PCA')}
'''
taxonomy_level 4-7 , taxnomy_group : sub PCA, mean, sum , epsilon: 0-1 
z_scoring: row, col, both, No, 'pca': (0, 'PCA') second element always PCA. first is 0/1 
normalization: log, relative , norm_after_rel: No, relative
'''
def preprocess(df, tag=None, taxonomy_level=7,taxnomy_group='mean',epsilon=0.1,normalization='log',z_scoring='No',norm_after_rel='No',pca= (0, 'PCA')):
    params={'taxonomy_level':taxonomy_level,'taxnomy_group':taxnomy_group, 'epsilon':epsilon, 'normalization':normalization,'z_scoring':z_scoring,'norm_after_rel':norm_after_rel,'pca':pca}

    if tag is None:
        mapping_file = CreateOtuAndMappingFiles(df, tags_file=None)
    else:
        mapping_file = CreateOtuAndMappingFiles(df,tag)
    mapping_file.preprocess(preprocess_params=params, visualize=0)

    return mapping_file.otu_features_df