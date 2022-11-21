import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub("titanic_batch_daily")
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","scikit-learn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("scalable"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")
    
    feature_view = fs.get_feature_view(name="titanic_modal", version=1)
    batch_data = feature_view.get_batch_data()
    
    #here we take the last uploaded passenger and we predict its survival state and save the predicted survival state
    y_pred = model.predict(batch_data)
    #print(y_pred)
    offset = 1
    passenger = y_pred[y_pred.size-offset]
    print("Passenger predicted: ", passenger)
    
    passenger_path =  "../img/" + str(passenger) + ".png"
    img = Image.open(passenger_path, mode='r')
    img.save("./latest_passenger.png")
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./latest_passenger.png", "Resources/images", overwrite=True)
   
   #here we take the actual last inserted flower of the feature group and we save its image
    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
    df = titanic_fg.read()
    #print(df)
    label = int(df.iloc[-offset]["survived"])
    label_path =  "../img/" + str(label) + ".png"
    img = Image.open(label_path, mode='r')
    print("Passenger actual: ", label)
    img.save("./actual_passenger.png")
    dataset_api.upload("./actual_passenger.png", "Resources/images", overwrite=True)
    
    #write a dataframe with the date of last prediction
    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Titanic Passenger Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [passenger],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4) #take last 4 rows of our dataframe
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    print("Number of different passenger predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Survived', 'True Dead'],
                         ['Pred Survived', 'Pred Dead'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different passenger predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different iris flower predictions") 

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("titanic_batch_daily")
        with stub.run():
            f()