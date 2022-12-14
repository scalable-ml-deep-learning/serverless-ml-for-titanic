import os
import modal
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

iris_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv")

iris_fg = fs.get_or_create_feature_group(
    name="iris_modal",
    version=1,
    primary_key=["sepal_length","sepal_width","petal_length","petal_width"], 
    description="Iris flower dataset")
iris_fg.insert(iris_df, write_options={"wait_for_job" : False})
