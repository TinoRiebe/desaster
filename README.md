# Disaster Response Pipeline Project

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






### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

new terminal 
env|grep WORK
https://SPACEID-3001.SPACEDOMAIN
