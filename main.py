import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_data

import logging
logger = logging.getLogger(__name__)

def main():
    logger.info("Reading raw data")
    raw_data = pd.read_csv('data/accepted_2007_to_2018q4.csv')
    logger.info("Preprocessing data")
    filtered_data = preprocess_data(raw_data)
    logger.info("Data preprocessing completed")
    
if __name__ == "__main__":
    main()

