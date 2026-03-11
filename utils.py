import pandas as pd
import numpy as np

def read_csv_with_header(filename):
    # Read the file and separate header and data
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Extract header (lines starting with "#")
    header_lines = [line[2:].strip() for line in lines if line.startswith("#")]
    data_start = next(i for i, line in enumerate(lines) if not line.startswith("#") and line.strip())

    # Read data using pandas
    df = pd.read_csv(filename, skiprows=data_start)

    return header_lines, df

def npz_to_csv(path):
    out_csv = path.replace(".npz", ".csv")
    data = np.load(path, allow_pickle=True)
    
    z, ra, dec = data['z'], data['ra'], data['dec']
    dataframe = pd.DataFrame({'RA': ra, 'DEC': dec, 'Z': z})
    dataframe.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")