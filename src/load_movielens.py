import csv
import numpy as np
import os

def load_movielens_data(data_folder_path):
    data_file = os.path.join(data_folder_path, 'u.data')

    my_data = np.zeros((943, 1682))

    file = np.genfromtxt(data_file, delimiter="\t", dtype=int)
    for row in file:
        my_data[row[0] - 1, row[1] - 1] = row[2]

    return my_data
