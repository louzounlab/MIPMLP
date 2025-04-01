import pandas as pd
from sklearn.decomposition import PCA

"""
Applies Principal Component Analysis (PCA) to reduce the dimensionality of the input data.

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
    # Fit PCA on the input data
    pca.fit(data)
    # Transform the data using the fitted PCA
    data_components = pca.fit_transform(data)
    # Generate a string summarizing explained variance per PCA component
    str_to_print = str("Explained variance per component: \n" +
          '\n'.join(['Component ' + str(i) + ': ' +
                     str(component) + ', Accumalative variance: ' + str(accu_var) for accu_var, (i, component) in zip(pca.explained_variance_ratio_.cumsum(), enumerate(pca.explained_variance_ratio_))]))

    str_to_print += str("\nTotal explained variance: " + str(pca.explained_variance_ratio_.sum()))
    # Print the variance summary to the console
    print(str_to_print)

    # Print the variance summary to the console
    return pd.DataFrame(data_components).set_index(data.index), pca, str_to_print


