# Disaster Response Pipeline Project
The three major aspects of this project is as follows:

ETL Pipeline - Clean,Transform and load data to Sqlite database
ML Pipeline - Build ML Pipeline
Flask web-app displaying analysis from data

A web app is created with Flask and Bootstrap for Natural Language Processing (NLP). The app provides an interface for new messages, (e.g. Twitter messages scanned by disaster relief agencies in a Disaster Response situation). Whenever you type a message it is classified into 37 Categories based on the learnings from the trained dataset

## Requirements
see requirement.txt

## File Structure
data folder:

disaster_categories.csv
disaster_messages.csv
DisasterResponse.db: the merge of cleaned messages and categories

proccess_data.py: contains the scripts to run etl pipeline for cleaning and saving the data

ML model folder contains the following:

ml_pipeline.py: contains scripts that create ml pipeline
disaster_resonse.pkl: contains the Classifier pickle fit file
train_classifier.py: script to train_classifier.py
app folder contains the following:

templates: Folder containing
index.html: Renders homepage
go.html: Renders the message classifier
run.py: Defines the app routes
img folder contains snapshots taken from web app:


## Description:
### process_data.py:

load, merge, clean and save the raw files from csv to sql

### train_classifier:

load the database and train the model. You can try a basic classifier without any additional parameters or use the predifined classifier
(SGDClassifier(loss='modified_huber', penalty='elasticnet', n_jobs=-1).
Finally, the trained modell will be saved

### run.py:

Test your modell on a webpage

INSTALLATION
Clone Repo
Rerun Scripts
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves

python ML model/train_classifier.py data/DisasterResponse.db ML model/disaster_response.pkl
Run the following command in the app's directory to run your web app. `python3 app/run.py

Go to http://0.0.0.0:3001/

RESULTS
Results are as follows:

The weighted avg precision and F1 score obtained from a test run using SGDClassifier are 0.75 and 0.61 respectively
The weighted avg precision and F1 score obtained from a test run using RandomForestClassifier 0.75 and 0.57 respectively
The weighted avg precision and F1 score obtained from a test run using GradientBoostingClassifier 0.70 and 0.60 respectively


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    
    `python run.py`
    
     better you run `python app/run.py` from the root directory

3. Go to http://0.0.0.0:3001/



python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

new terminal 
env|grep WORK
https://SPACEID-3001.SPACEDOMAIN
