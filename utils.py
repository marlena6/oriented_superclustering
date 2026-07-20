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

# computing the distance to the nearest edge of the map 
def dist_to_nearest_edge(dec_rad, ra_rad, dec_min_map,dec_max_map,ra_min_map,ra_max_map):
    """Minimum angular distance [rad] from (dec, ra) to the map boundary."""
    cos_dec = np.cos(dec_rad)
    return np.minimum.reduce([
        dec_max_map - dec_rad,          # to north edge
        dec_rad   - dec_min_map,        # to south edge
        cos_dec * (ra_max_map - ra_rad),  # to east edge
        cos_dec * (ra_rad - ra_min_map),  # to west edge
    ])
