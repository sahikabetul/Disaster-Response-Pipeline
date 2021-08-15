# Disaster Response Pipeline Project
**
###Project Overview:**
In this project, I applied data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

My project includes a web app where an emergency worker can input a new message and get classification results in several categories:

![result](/Classification_Results.PNG?raw=true "Classification result")

The web app also displays two visualizations of the data:

![visualizations](/Data_Visualizations.PNG?raw=true "Visualizations of data")

The structure of the project:
![structure](/files.png?raw=true "Visualizations of data")

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Dependencies:
numpy, pandas, matplotlib, json, plotly, nltk, flask, sklearn, sqlalchemy, sys, re, pickle

This project also requires Python 3.x along with the above libraries installed as a pre-requisite
