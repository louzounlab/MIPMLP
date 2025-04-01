import MIPMLP
import pandas as pd

df = pd.read_csv("example_input_files/example_input_files1/OTU.csv")

df1 = MIPMLP.preprocess(df, plot=True)

df1.to_csv("OTU_MIP.csv", index=False)

