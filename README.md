# Disaster Response Pipeline Project

### Introduction

In this project I built an ETL machine learning pipeline that can classify (twitter) messages into several categories of disasters. This project also features a webapp (flask) where the data is made visual with plotly and where you can classify your own messages. 

### File descriptions

data/process_data.py:  The ETL pipeline used to load and clean data as well as saving it to a database
models/train_classifier.py: The Machine Learning pipeline used to fit, optimalize, evaluate, and export the model to a Python pickle file
app/templates/~.html: HTML pages for the web app
run.py: Start the Python server for the web app


### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run the ETL pipeline that cleans data and stores it in a database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains a random forest classifier and saves it as a pickle file
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:8080/
