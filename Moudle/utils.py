# -*- coding: utf-8 -*-
# @Time    : 2023/2/27 16:16
# @Author  : hxt
# @FileName: utils.py
# @Software: PyCharm


# importing pandas as pd
import pandas as pd


# read an excel file and convert
# into a dataframe object

# show the dataframe
# importing pandas as pd
# Read and store content
# of an excel file

def convert_to_csv(path):

    read_file = pd.read_excel(path)
    des_path=path.split("xlsx")[0]+"csv"
    print(des_path)
    # Write the dataframe object
    # into csv file
    read_file.to_csv(des_path,
                     index=None,
                     header=True)

    # read csv file and convert
    # into a dataframe object

def main():
    csv_path="../Data/smile.xlsx"
    print(csv_path)
    read_file = pd.read_excel(csv_path)
    des_path = csv_path.split("xlsx")[0] + "csv"
    print(des_path)
    read_file.to_csv(des_path,
                     index=None,
                     header=True)

