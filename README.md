# MIPMLP
## Preprocessing for 16S values
Preproceesing the raw ASVs by the following steps: 
(1) merging similar features based on the taxonomy, (2) scaling the distribution, (3) standardization to z-scores, (4) and dimension reduction.

### Input    
The input file for the preprocessing should contain detailed unnormalized OTU/Feature values as a biom table, the appropriate taxonomy as a tsv file, 
**OR** 
a merged otu and taxonomy csv, such that the first column is named "ID", each row represents a sample and each column represents an ASV. The last row contains the taxonomy information, named "taxonomy".
Here is an example of how the input OTU file should look like : ([file example](https://mip-mlp.math.biu.ac.il/download-example-files))

<img src="https://drive.google.com/uc?export=view&id=18s12Zxc4nOHjk0vr8YG8YQGDU0D8g7wp" alt="drawing" width="400" height="450"/>

#### Optional
It is possible to load a **tag file**, with the class of each sample.
The tag file is not used for the preprocessing, but is used to provide some statistics on the relation between the features and the class.
**Note: One can also run the preprocessing without a tag file.** 

### Preprocessing Parameters
Now you will have to select the parameters for the preprocessing.
1. **The taxonomy level used** - taxonomy sensitive dimension reduction by grouping the taxa at
 a given taxonomy level. All features with a given representation at a given taxonomy
 level will be grouped and merged using three different methods: Average, Sum or Sub-PCA (using PCA then followed by normalization).
2. **Normalization** - after the grouping process, you can apply two different normalization methods. the first one is the log (10 base) scale. in this method 
x → log10(x + ɛ),where ɛ is a minimal value to prevent log of zero values. 
The second methos is to normalize each bacteria through its relative frequency.
        If you chose the **log normalization**, you now have four standardization 
                a) No standardization
                b) Z-score each sample
                c) Z-score each bacteria
                d) Z-score each sample, and Z-score each bacteria (in this order)

    When performing **relative normalization**, we either do not standardize the results
    or performe only a standardization on the taxa.
    
3. Dimension reduction - after the grouping, normalization and standardization you can choose from two Dimension reduction method: PCA or ICA. If you chose to apply a Dimension reduction method, you will also have to decide the number of dimensions you want to leave.

### How to use
```python
MIPMLP.preprocess(input_df)
```
#### Parameters:
-**taxonomy_level**: 4-7 , default is 7
-**taxnomy_group** : sub PCA, mean, sum, default is mean
-**epsilon**: 0-1
-**z_scoring**: row, col, both, No, default is No
-**pca**: (0, 'PCA') second element always PCA. first is 0/1
-**normalization**: log, relative, default is log
-**norm_after_rel**: No, relative, default is No

### Output
The output is the processed file.

<img src="https://drive.google.com/uc?export=view&id=1UPdJfUs_ZhuWFaHmTGP26gD3i2NFQCq6" alt="drawing" width="400" height="400"/>

## iMic 
 iMic is a  method to combine information from different taxa and improves data representation for machine learning using microbial taxonomy. 
iMic translates the microbiome to images, and convolutional neural networks are then applied to the image.

### micro2matrix
Translates the microbiome values and the cladogram into an image. **micro2matrix** also saves the images that were created in a given folder.
#### Input

-**df** A pandas dataframe which is similar to the MIPMLP preprocessing's input (above).
-**folder** A folder to save the new images at.

#### Parameters
You can determine all the MIPMLP preprocessing parameters too, otherwise it will run with its deafulting parameters (as explained above).

#### How to use
```python
import pandas as pd
df = pd.read_csv("address/ASVS_file.csv")
folder = "save_img_folder"
MIPMLP.micro2matrix(df, folder)
```
	
### CNN2 class - optional
A model of 2 convolutional layer followed by 2 fully connected layers.

####CNN model parameters
-**l1 loss** = the coefficient of the L1 loss
-**weight decay** = L2 regularization
-**lr** = learning rate
-**batch size** = as it sounds
-**activation** = activation function one of:  "elu", | "relu" | "tanh"
-**dropout** = as it sounds (is common to all the layers)
-**kernel_size_a** = the size of the kernel of the first CNN layer (rows)
-**kernel_size_b** = the size of the kernel of the first CNN layer (columns)
-**stride** = the stride's size of the first CNN
-**padding** = the padding size of the first CNN layer
-**padding_2** = the padding size of the second CNN layer
-**kernel_size_a_2** = the size of the kernel of the second CNN layer (rows)
-**kernel_size_b_2** = the size of the kernel of the second CNN layer (columns)
-**stride_2** = the stride size of the second CNN
-**channels** = number of channels of the first CNN layer
-**channels_2** = number of channels of the second CNN layer
-**linear_dim_divider_1** = the number to divide the original input size to get the number of neurons in the first FCN layer
-**linear_dim_divider_2** = the number to divide the original input size to get the number of neurons in the second FCN layer
-**input dim** = the dimention of the input image (rows, columns)

#### How to use
	params = {
        "l1_loss": 0.1,
        "weight_decay": 0.01,
        "lr": 0.001,
        "batch_size": 128,
        "activation": "elu",
        "dropout": 0.1,
        "kernel_size_a": 4,
        "kernel_size_b": 4,
        "stride": 2,
        "padding": 3,
        "padding_2": 0,
        "kernel_size_a_2": 2,
        "kernel_size_b_2": 7,
        "stride_2": 3,
        "channels": 3,
        "channels_2": 14,
        "linear_dim_divider_1": 10,
        "linear_dim_divider_2": 6,
		"input_dim": (8,100)
    }
    model = MIPMLP.CNN(params)

A trainer on the model should be applied by the user after choosing the best hyperparameters by an [NNI](https://nni.readthedocs.io/en/stable/) platform.

### apply_iMic (a basic example run of iMic function)
A basic running iMic option of uploading the images dividing them to a training set and test set and returns the real labels (train and test) and the predicted labels (train and test)

#### Input
-**tag** A tag pandas dataframe with similar samples to the raw ASVs file.
-**folder** A folder of the saved images from the micro2matrix step.
-**test_size** Fraction of the test set from the whole cohort (default is 0.2).
-**params** iMic model's hyperparameters. Should be selected for each dataset separately by grid-search or [NNI](https://nni.readthedocs.io/en/stable/) on appropriate
validation set. The default params are
{
    "l1_loss": 0.1,
    "weight_decay": 0.01,
    "lr": 0.001,
    "batch_size": 128,
    "activation": "elu",
    "dropout": 0.1,
    "kernel_size_a": 4,
    "kernel_size_b": 4,
    "stride": 2,
    "padding": 3,
    "padding_2": 0,
    "kernel_size_a_2": 2,
    "kernel_size_b_2": 7,
    "stride_2": 3,
    "channels": 3,
    "channels_2": 14,
    "linear_dim_divider_1": 10,
    "linear_dim_divider_2": 6,
    "input_dim": (8, 235)
})

**Note that the input_dim is also updated automatically during the run.**

#### Output
A dictionary of {"pred_train": pred_train,"pred_test": pred_test,"y_train": y_train,"y_test": y_test}

#### How to use
```python
# Load tag
tag = pd.read_csv("data/ibd_tag.csv", index_col=0)

# Prepare iMic images
otu = pd.read_csv("data/ibd_for_process.csv")
MIPMLP.micro2matrix(otu, folder="data/2D_images")

# Run a toy iMic model. One should optimize hyperparameters before
dct = apply_iMic(tag, folder="data/2D_images")
```




# Citation

- Shtossel, Oshrit, et al. "Ordering taxa in image convolution networks improves microbiome-based machine learning accuracy." Gut Microbes 15.1 (2023): 2224474.

- Jasner, Yoel, et al. "Microbiome preprocessing machine learning pipeline." Frontiers in Immunology 12 (2021): 677870.

