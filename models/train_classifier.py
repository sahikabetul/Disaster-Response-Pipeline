import sys
import pandas as pd
import numpy as np
import re
import pickle

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn. ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, accuracy_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """Load the specific database
    
    Load the data from the specific file.
    
    Parameters:
    -----------
    database_filepath: string
        File path of the specific database
    
    Results:
    ----------
    X: ndarray
        Ndarray values store the message information
    Y: ndarray
        Ndarray values store the the multi class information
    category_names: list
        List contains the multi class string
    """
    # loading data from .db file
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(database_filepath, engine)
    
    # creating input and label lists
    X = df.id.values
    Y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    #cleaning url's from text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    #normalizing text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) #[^a-zA-Z0-9]
    text = text.lower()    

    #tokenizing text
    tokens = word_tokenize(text)
    
    #lemmatizing and removing stop words
    lemmatizer = WordNetLemmatizer()
    
    # listing tokens without stop-words
    stop_words = stopwords.words("english")
    clean_tokens = []
    for token in tokens:
        if token not in stop_words:
            clean_tokens.append(lemmatizer.lemmatize(token))
    
    return clean_tokens


def build_model():
    #creating model
    forest = RandomForestClassifier(random_state=42, n_jobs=4)

    #creating pipeline for preprocesses and training
    pipeline = Pipeline([
        ("text_pipeline", Pipeline([
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer())
        ])),
        ("clf", MultiOutputClassifier(forest, n_jobs=2))
    ])
    
    # define parameters for grid search
    parameters = {
    'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    'text_pipeline__vect__max_df': (0.75, 1.0),
    'text_pipeline__vect__max_features': (5000, 10000),
    'clf__estimator__n_estimators': [10, 20],
    'clf__estimator__max_depth': [4, 6],
    }
    
        # choose a method to build model
    print("Hint: GridSearchCV will takes more time!\n")
    chose_option = input("Do you want to GridSearchCV(Yes or No): ").lower()
    while True:
        if chose_option in ["yes", "y"]:
            model = GridSearchCV(pipeline, param_grid=parameters, cv=3)
        elif chose_option in ["no", "n"]:
            model = pipeline
        else:
            print("Choose a validate option!")
            model = False
        
        if model:
            return model


def evaluate_model(model, X_test, Y_test, category_names):
    # predicting labels
    Y_pred = model.predict(X_test.astype(str))
    
    #comparing predicted labels with ground-truth
    for index, column in enumerate(category_names):
        report = classification_report(Y_test[:, index], Y_pred[:, index], labels=np.arange(1), target_names=[column],  digits=3)
    
    return report


def save_model(model, model_filepath):
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train.astype(str), Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test.astype(str), Y_test, category_names)

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