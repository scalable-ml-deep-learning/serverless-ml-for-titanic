# Serverless Machine Learning for Titanic Dataset

Contributors:
<a href="https://github.com/Bralli99">Brando Chiminelli</a>, 
<a href="https://github.com/boyscout99">Tommaso Praturlon</a>

Course: <a href="https://id2223kth.github.io/">Scalable Machine Learning and Deep Learning</a>, at <a href="https://www.kth.se/en">KTH Royal Institute of Technology</a>

## About

In this project we built a scalable serverless machine learning predictive system, consisting of a feature pipeline, a training pipeline, a batch inference pipeline, and a user interface (one for interactive querying, one as a dashboard).

In particular, we trained the logistic regression classifier model on the cleaned dataset of titanic survivals (https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv).

Our features are the passenger class, sex, age and fare he paid for the ticket and the target we want to predict is if the passenger survived or not.

## Implementation

The following are the steps we implemented:

1. We wrote a feature pipeline that registers the titantic dataset as a Feature Group with Hopsworks.
2. We wrote a training pipeline that reads the training data, which is 80% of our dataset, with a Feature View from Hopsworks, trains a logistic regression classifier model to predict if a particular passenger survived the Titanic or not. Then we registered the model with Hopsworks.
3. We wrote a Gradio application that downloads the model saved in Hopsworks and provides a user interface to allow users to enter or select feature values to predict if a passenger with the provided features would survive or not.
4. We wrote a synthetic data passenger generator and update the feature pipeline to allow it to add new synthetic passengers every 24h. This is done with Modal that allows us to run our titanic-feature-pipeline-daily on a serverless infrastructure every day.
5. We wrote a batch inference pipeline to predict if the last synthetic passengers insered in the Feature Group survived or not, and built a Gradio application to show the most recent synthetic passenger prediction and outcome, and a confusion matrix with historical prediction performance.
Also the batch inference pipeline runs on Modal every 24h.

## Spaces on Hugging Face

### Iris - Spaces
https://huggingface.co/spaces/bralli/iris

### Iris Monitor - Spaces
https://huggingface.co/spaces/bralli/iris-monitor

### Titanic - Spaces
https://huggingface.co/spaces/bralli/titanic

### Titanic Monitor - Spaces
https://huggingface.co/spaces/bralli/titanic-monitor

## Built With

* [Hopsworks](https://app.hopsworks.ai/)
* [Modal](https://modal.com/)
* [Hugging Face](https://huggingface.co/)

