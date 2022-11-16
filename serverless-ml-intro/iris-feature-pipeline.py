import os
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   #image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("scalable"))
   def f():
       g()

def g():
    import hopsworks
    import pandas as pd
    
    file_key = open("./ENV_VARS/HOPSWORKS_API_KEY", "r")
    HOPSWORKS_API_KEY = file_key.read()
    print(HOPSWORKS_API_KEY)
    project = hopsworks.login(port=443, api_key_value=HOPSWORKS_API_KEY.strip())
    fs = project.get_feature_store()
    iris_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv")
    iris_fg = fs.get_or_create_feature_group(
        name="iris_modal",
        version=1,
        primary_key=["sepal_length","sepal_width","petal_length","petal_width"], 
        description="Iris flower dataset")
    iris_fg.insert(iris_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
