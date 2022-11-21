import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def passenger(pclass, sex, age, fare,):
    input_list = []
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(age)
    input_list.append(fare)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    print(res)
    passenger_url = "https://raw.githubusercontent.com/scalable-ml-deep-learning/serverless-ml-for-titanic/feature-brando/img/" + str(res[0]) + ".png"
    img = Image.open(requests.get(passenger_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=passenger,
    title="Passenger Survival Predictive Analytics",
    description="Experiment with passenger class, fare, sex and age to predict if the passenger survived.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="Passenger class"),
        gr.inputs.Number(default=1.0, label="Passenger sex (1:male, 0:female)"),
        gr.inputs.Number(default=1.0, label="Passenger age"),
        gr.inputs.Number(default=1.0, label="Passenger fare")
        ],
    outputs=gr.Image(type="pil"))

demo.launch()