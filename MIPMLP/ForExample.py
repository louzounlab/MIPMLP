import MIPMLP
import pandas as pd

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

# Run the pipeline
df_train_processed, df_test_processed = MIPMLP.preprocess(train_df_with_last, df_test=test_df_with_last, plot=True, test_flag=True)


# Save processed output
df_train_processed.to_csv("OTU_MIP_train.csv", index=False)
df_test_processed.to_csv("OTU_MIP_test.csv", index=False)