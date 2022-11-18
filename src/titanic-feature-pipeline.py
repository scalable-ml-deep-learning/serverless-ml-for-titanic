import os
import modal
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

titanic_df = pd.read_csv("../data/titanic.csv")

titanic_df.head()

'''
titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    version=1,
    primary_key=["sepal_length","sepal_width","petal_length","petal_width"], 
    description="Titanic survival dataset")
titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})
'''
