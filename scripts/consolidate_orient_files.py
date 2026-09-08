# When you have run separate orient files at different redshifts, consolidate them into one catalog
import numpy as np
import pandas as pd
import sys

outfile = sys.argv[1]
files = sys.argv[2:]

total_dataframe = pd.DataFrame()
for file in files:
    if not file.endswith('.csv'):
        sys.exit("Failure; one of the files passed was not a .csv file. Please pass only .csv files.")
    data = pd.read_csv(file, comment='#')
    total_dataframe = pd.concat([total_dataframe, data], ignore_index=True)

total_dataframe.to_csv(outfile, index=False, mode='x')
print("Saved consolidated catalog to", outfile)