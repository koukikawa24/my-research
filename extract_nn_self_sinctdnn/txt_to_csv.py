#txt_to_csv.py

import pandas as pd
import os
import re 
# read_text_file = pd.read_csv (r"label/training_age_sex.txt",sep= " ")
# read_text_file.to_csv (r"label/training_age_sex.csv", index=None)

read_text_file = pd.read_csv (r"label/training_age_sex.csv")

path1 = read_text_file[0] 
paht2 = "wav"

read_text_file[0]