# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:30:17 2023

@author: Sebastian
"""

DATA_DIRECTORY = "data/"
OUTPUT_DIRECTORY = "output/"

import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

food_data = pd.read_csv(DATA_DIRECTORY + "food_data.csv")

with(open(DATA_DIRECTORY + "nutrients.txt", "r")) as nutrient_file:
    nutrients = nutrient_file.read().splitlines()

#Choosing not to impute - 1197 rows is ~15%
XY = food_data[nutrients + ["nutrition_grade_fr"]].dropna()

X = XY[nutrients]
Y = XY["nutrition_grade_fr"]

plt.figure()
plt.title = "Test"
# X.plot(kind = "hist", title = "Test", 
#        sharey = False, sharex = False, subplots = True, 
#        layout = (1,len(X.columns)))

#food_data.hist(column = nutrients, by = "food_category", sharex = False, sharey = False)
#%% seaborn histogram variables
hist_data = pd.melt(food_data[nutrients + ["food_category"]].dropna(), id_vars = "food_category")
g = sb.FacetGrid(hist_data, col="variable", row="food_category", 
                 margin_titles=True, sharex = False, sharey = False)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.map(sb.histplot, "value", bins=30)
plt.savefig(OUTPUT_DIRECTORY + "nutrient_food_distributions.png", dpi = 300)

#%% seaborn histogram nutriscore
hist_data = food_data[["nutrition_grade_fr", "food_category"]].dropna()
hist_data['nutrition_grade_fr'] = hist_data['nutrition_grade_fr'].astype('category')
hist_data['nutrition_grade_fr'].cat.reorder_categories(['a', 'b', 'c', 'd', 'e'])


g = sb.FacetGrid(hist_data, col="food_category", 
                 margin_titles=True, sharex = False, sharey = False)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.map(sb.histplot, "nutrition_grade_fr")
plt.savefig(OUTPUT_DIRECTORY + "nutriscore_food_distributions.png", dpi = 300)

#%% pyplot histograms
# input variable (nutrient) plot
X.hist(sharey = False, sharex = False)
plt.tight_layout(w_pad = 0.01, h_pad = 0.01, pad = 0.01)
plt.gcf().set_size_inches(15,7.5)
plt.savefig(OUTPUT_DIRECTORY + "nutrient_distributions.png", dpi = 300)
g.map(sb.histplot, "value", bins=30)
# nutriscore plot
Y.hist()
plt.savefig(OUTPUT_DIRECTORY + "nutriscore_distributions.png", dpi = 300)
