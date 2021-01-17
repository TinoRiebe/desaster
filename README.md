# Disaster Response Pipeline Project
Overview: vlablavlub

## Requirements
Python 3.x

process_data_py:
- pandas
- sqlalchemy

train_classifier.py:
- pandas
- sqlalchemy
- re
- nltk
- sklearn
- joblib

run.py:
- plotly
- nltk
- json
- sqlalchemy
- joblib

## Description:
### process_data.py:

load, merge, clean and save the raw files from csv to sql

### train_classifier:

load the database and train the model. You can try a basic classifier without any additional parameters or use the predifined classifier
(SGDClassifier(loss='modified_huber', penalty='elasticnet', n_jobs=-1).
Finally, the trained modell will be saved

### run.py:

Test your modell on a webpage




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
