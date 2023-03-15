# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:10:08 2023

@author: Sebastian
"""

import os
from zipfile import ZipFile
import pandas as pd
from pathlib import Path


RAW_DATA_DIR = "raw_data/"
DATA_DIRECTORY = "data/"

Path(DATA_DIRECTORY).mkdir(parents=True, exist_ok=True)

os.system(f"kaggle datasets download -d openfoodfacts/world-food-facts -p {RAW_DATA_DIR}")

#Using a zip so we don't have to have a 1GB file on our hard drive
food_zip = ZipFile(RAW_DATA_DIR + "world-food-facts.zip", 'r')
raw_fooddata = pd.read_table(food_zip.open("en.openfoodfacts.org.products.tsv"), encoding="utf-8")

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
food_data = raw_fooddata[raw_fooddata.food_category != "other"]

food_data.to_csv(DATA_DIRECTORY + "food_data.csv", index = False)
