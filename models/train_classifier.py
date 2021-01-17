# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 18:03:49 2021

@author: Tino Riebe

"""
# Basics
import os
import pandas as pd
from sqlalchemy import create_engine
import re
import sys 


# Language Toolit
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# klearn Libraries for Ml Models
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib as jbl

# sklearn classifier 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


import warnings
warnings.filterwarnings("ignore");

# database_filepath = 'e:/github/desaster/data/DisasterResponse.db'
# model_filepath = 'e:/github/desaster/models'

def load_data(database_filepath):
    '''
    INPUT  
    database_filepath(string) - DisasterResponse.db 
        
    OUTPUT
    X - features the datset
    y - target of the dataset
    categories - the name of the different targets
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Disaster_messages', engine)
    engine.dispose()
    
    X = df['message']
    y = df[df.columns[4:]]
    categories = y.columns.tolist()

    return X, y, categories


def tokenize(text):
    """
    INPUT
    text - messages column from the table
    
    OUTPUT
    lemmed - tokenized text after performing below actions
    
    1. Remove Punctuation and normalize text
    2. Tokenize text and remove stop words
    3. Use stemmer and Lemmatizer to Reduce words to its root form
    """
    # Remove Punctuations and normalize text by converting text into lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text and remove stop words
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    words = [w for w in tokens if w not in stop_words]
    
    #Reduce words to its stem/Root form
    stemmed = [PorterStemmer().stem(w) for w in words]
    # Lemmatize verbs
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in stemmed]
    
    return lemmed


def build_basic_pipelines(clf_name):
    """
    INPUT
    clf_name - name of a classifier
    
    OUTPUT
    pipeline - basic pipeline with classifier
    """
    
    clf_dict={'RandomForest':1, 'KNeighbors':2,'GradientBoosting':3,
              'DecisionTree':4, 'AdaBoost':5, 'SGD':7,
              'MultinominalNB':8, 'SVC':9}
    
    if clf_name not in clf_dict:
        print('unsupported classifier, please choose only:\n')
        for clf in clf_dict.keys():
            print(clf)
    else:
        print(clf_name, '- pipeline will be prepared')
    
    if clf_dict[clf_name]==1:
        clf = RandomForestClassifier()
    elif clf_dict[clf_name]==2:
        clf = KNeighborsClassifier()
    elif clf_dict[clf_name]==3:
        clf = GradientBoostingClassifier()
    elif clf_dict[clf_name]==4:
        clf = DecisionTreeClassifier()
    elif clf_dict[clf_name]==5:
        clf = AdaBoostClassifier()
    elif clf_dict[clf_name]==6:
        clf = SGDClassifier()
    elif clf_dict[clf_name]==7:
        clf = MultinomialNB()
    elif clf_dict[clf_name]==8:
        clf = SVC()
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(clf))
        ])
    
    return pipeline


def build_model():
    """
    INPUT
    None - because we will use the selectd classifier with some parameters
    
    OUTPUT
    cv - the pipeline
    """
    # The pipeline has tfidf, dimensionality reduction, and classifier
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier(loss='modified_huber',
                                                    penalty='elasticnet',
                                                    n_jobs=-1))),
        ])

    return pipeline   


def evaluate_model(model, X_test, y_test, categories):
    """
    INPUT:
    model: ml -  model
    X_text: The X test set
    y_test: the y test classifications
    category_names: the category names
    
    OUTPUT:
    None
    """
        
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=categories))


def save_model(model, model_filepath):
    """
    INPUT: 
    model - the fitted model
    model_filepath (str) -  filepath to save model
    
    OUTPUT:
    None
    """
    
    
    jbl.dump(model, model_filepath)
    print('modell is saved:', model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')





if __name__ == '__main__':
    main()
