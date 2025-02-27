import pandas as pd
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