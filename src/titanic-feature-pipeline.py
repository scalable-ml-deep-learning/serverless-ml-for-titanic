import os
import modal
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

titanic_df = pd.read_csv("../data/titanic.csv")

print(titanic_df.head())
titanic_df.dropna(inplace=True)
titanic_df.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
titanic_df.Sex.replace({'male':1, 'female':0}, inplace=True)
titanic_df.rename(columns={"Pclass": "pclass", "Sex": "sex", "Fare": "fare", "Age": "age"})
print(titanic_df.head())


titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    version=1,
    primary_key=["pclass","fare","sex","age"], 
    description="Titanic survival dataset")
titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

