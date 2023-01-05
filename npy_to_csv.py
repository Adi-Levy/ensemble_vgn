# for every file in the sim_res/{internal_dir} directory, load the npy data and add it to a new sheet in a csv file corresponding to that round_id and save to the same directory as the npy files. The csv file should have the same name as the npy file, but with a .csv extension.

import os
import numpy as np
import pandas as pd
import XlsxWriter

def npy_to_csv(npy_dir):
    for file in os.listdir(npy_dir):
        if file.endswith(".npy"):
            file_name = file.split(".")[0]
            _, round_id, attempt = file_name.split("_")
            npy_data = np.load(file)
            df = pd.DataFrame(npy_data, columns = ['pos_x','pos_y','pos_z','rot_x','rot_y','rot_z','width','score'])
            # Create a Pandas Excel writer using XlsxWriter as the engine.
            writer = pd.ExcelWriter(f'round_{round_id}.xlsx', engine='xlsxwriter')
            # Convert the dataframe to an XlsxWriter Excel object.
            df.to_excel(writer, sheet_name=f'attempt_{attempt}')
            # Close the Pandas Excel writer and output the Excel file.
            writer.save()