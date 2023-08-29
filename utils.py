import pandas as pd


def getData(path):
    """This function loads the data from csv file"""
    data = pd.read_csv(path)
    return data
