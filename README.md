# Disaster Response Project

## Introduction
This is a project designed to categorize disaster events so that you can send the messages to an appropriate disaster relief agency using a data set containing real messages that were sent during disaster events.

It includes three major phases:
1. **ETL Pipeline**: An ETL pipeline where it loads the messages and categories datasets, merge them together, clean the output and store it in a SQLite database
2. **ML Pipeline**: An machine learning pipeline where it loads data from the SQLite database, splits the dataset into training and test sets, builds a text proc essing, train and tune using GridSearchCV and export the output as pickle file
3. **Flask Web App**: a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## How to use:
1. Install all dependencies using:
    - `pip install -r requirements.txt`
2. To run the ETL pipeline
    - `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
3. To run the ML pipeline
    - `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
4. to run the web app
    - `python run.py`
    - Go to http://0.0.0.0:3001/

## Structure
```
- app
| - static
| |- css # static styling files
| |- images # images
| |- |- graphs.png
| |- js # external javascript files
| - templates
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
- .gitignore
- requirements
```

## Graphs
![Alt text](app/static/images/graphs.png?raw=true "Title")


## Authors
Mostafa Abdelrashied

## Acknowledgments
- SmokeShine: His code give me some inspirations to fasten my training process
    - https://github.com/SmokeShine/disaster_response_pipeline_project
