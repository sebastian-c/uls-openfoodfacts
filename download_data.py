# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:10:08 2023

@author: Sebastian
"""

import os
import zipfile
import pandas as pd
from pathlib import Path


RAW_DATA_DIR = "raw_data/"
DATA_DIRECTORY = "data/"

Path(DATA_DIRECTORY).mkdir(parents=True, exist_ok=True)

os.system(f"kaggle datasets download -d openfoodfacts/world-food-facts -p {RAW_DATA_DIR}")

with zipfile.ZipFile("raw_data/world-food-facts.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")

raw_fooddata = pd.read_table(DATA_DIRECTORY + "en.openfoodfacts.org.products.tsv", encoding="utf-8")

## Important columns
# categories_tags
# nutrition_grade_fr
# 

## Categories
# en:biscuits-and-cakes
# en:breakfast-cereals
# en:iced-teas

def defineFoodCategory(series):
    series_str = str(series)
    categories = series_str.split(sep=",")
    if "en:breakfast-cereals" in categories:
        return("breakfast-cereal")
    elif "en:biscuits-and-cakes" in categories:
        return("biscuit-cake")
    elif "en:iced-teas" in categories:
        return("iced-tea")
    else: 
        return("other")

raw_fooddata.loc[:,"food_category"] = raw_fooddata.categories_tags.apply(defineFoodCategory)

raw_fooddata.to_csv(DATA_DIRECTORY + "food_data.csv", index = False)
