# Project Overview
A data science project that analyzed disaster data from Figure Eight to build a machine learning model that classifies disaster messages.

A machine learning pipeline is created to categorize events and requests during disaster so that the one can send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. 
The web app will also display visualizations of the data. 

<img src="/docs/dr.png" alt="screenshot"/>


# Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Important Files:

data/process_data.py: The ETL pipeline used to process data for model building.

models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle 

app/templates/*.html: HTML templates for the web app.

run.py: Start the Python server for the web app.
