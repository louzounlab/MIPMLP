import MIPMLP
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")



# Load the data
df = pd.read_csv("example_input_files/example_input_files1/OTU.csv")

# Separate the last row 'taxonomy'
last_row = df.tail(1)
df_main = df.iloc[:-1]  # All rows except the last

# Split the main data into 80% train and 20% test (random split)
train_df = df_main.sample(frac=0.8, random_state=42)
test_df = df_main.drop(train_df.index)

# Add the last row to both sets
train_df_with_last = pd.concat([train_df, last_row], ignore_index=True)
test_df_with_last = pd.concat([test_df, last_row], ignore_index=True)

# Save to CSV files
train_df_with_last.to_csv("OTU_train.csv", index=False)
test_df_with_last.to_csv("OTU_test.csv", index=False)



# --- Option 1: full pipeline with train and test (with sub pca) ---
# df_train_processed, df_test_processed = MIPMLP.preprocess(
#    train_df_with_last,
#    df_test=test_df_with_last,
#    plot=True,
#    taxnomy_group='sub PCA')
# #Save processed output
# df_train_processed.to_csv("OTU_MIP_train.csv", index=False)
# df_test_processed.to_csv("OTU_MIP_test.csv", index=False)



# --- Option 2: single dataset with external sub PCA ---
#with open("sub_pca_scaler.pkl", "rb") as f:
  #  saved_sub_pca = pickle.load(f)

  #  df_single_processed = MIPMLP.preprocess(
   #     train_df_with_last,  # using train as a single dataset
    #    external_sub_pca=saved_sub_pca,
    #    taxnomy_group='sub PCA',
   #     plot=True
     #   )
#df_single_processed.to_csv("OTU_MIP_single.csv", index=False)




# --- Option 3: full pipeline with train and test (with pca) ---
#df_train_processed, df_test_processed = MIPMLP.preprocess(
#   train_df_with_last,
  # df_test=test_df_with_last,
  # pca= (1, 'PCA'))

# Save processed output
#df_train_processed.to_csv("OTU_MIP_train.csv", index=False)
#df_test_processed.to_csv("OTU_MIP_test.csv", index=False)



# --- Option 4: single dataset with external PCA ---
# with open("pca_scaler.pkl", "rb") as f:
#     saved_pca = pickle.load(f)
#
#     df_single_processed = MIPMLP.preprocess(
#         train_df_with_last,  # using train as a single dataset
#         external_pca=saved_pca,
#         taxnomy_group='sub PCA',
#         )
# df_single_processed.to_csv("OTU_MIP_single.csv", index=False)


# --- Option 5: single database (8 taxomony) without external or_preprocess_8_taxonoscaler ---
df1 = pd.read_csv("example_input_files/for_preprocess_8_taxonomy.csv")
df_single_processed = MIPMLP.preprocess(
       df1,
       taxnomy_group='sub PCA',
       taxonomy_level=8,
       drop_tax_prefix=False,
       plot=True
       )
df_single_processed.to_csv("OTU_MIP_single_8.csv", index=False)
