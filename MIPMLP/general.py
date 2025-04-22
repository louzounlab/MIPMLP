import pandas as pd
from sklearn.decomposition import PCA

"""
Applies PCA to reduce the dimensionality of the input data.

Parameters:
- data: DataFrame of input features.
- n_components: Number of PCA components to keep.

Returns:
- A DataFrame of transformed data in the reduced dimension space.
- The PCA object used for the transformation.
- A string summary of explained variance per component.
"""


def apply_pca(data, n_components=15):
    # Initialize the PCA object with the desired number of components
    pca = PCA(n_components=n_components)
    # Fit PCA on the input data and transform it
    data_components = pca.fit_transform(data)

    # Return the transformed data and the fitted PCA object
    return pd.DataFrame(data_components, index=data.index), pca
