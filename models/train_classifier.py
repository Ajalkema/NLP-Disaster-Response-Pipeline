
# import libraries
import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import joblib
from datetime import datetime

# import and download NLP packages and vocabulary
import nltk
import re
nltk.download(['punkt', 'wordnet'])

# import SKLearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, recall_score, precision_score



def load_data(database_filepath):
    '''
    Loads an sql database into a Pandas dataframe. Seperates 
    the messages column into the X variable and puts the 
    message categories into the Y variable. The column names
    of the Y variable are put into the category_names variable.
    
    Args: 
        database_filepath: the file path to the database file
    
    Returns: 
        X: Dataframe with the messages
        Y: Dataframe with the categories
        category_names: list with the category names 
    '''

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Messages', engine)
    
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    '''
    Replaces URL's found in the text with a placeholder name.
    Tokenizes and then lemmatizes each token. All text is made
    lowercase and leading and trailing whitespace is removed. 

    Args:
        text: The text to be tokenized/lemmatized
    
    Returns: 
        clean_tokens: List with the lemmatized tokens
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    urls_found = re.findall(url_regex, text)
    for url in urls_found:
        text = text.replace(url, "urlplaceholder")
    
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    Creates the model using GridSearchCV. Gridsearch finds the best estimator 
    and parameters out of two estimators, Random Forest Classificator and 
    Linear Support Vector Machine, and several parameters for each estimator.

    Args:
        None

    Returns:
        cv: The model with optimized parameters
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # According to GridSearch:
    # Best parameters:  Best estimator=LinearSVC
    # clf__estimator__C': 0.75, 'clf__estimator__penalty': 'l1', 
    # 'tfidf__use_idf': False, 'vect__max_df': 0.7
    parameters = {
        'vect__max_df':[0.4, 0.7, 1.0],
        'tfidf__use_idf': [True, False],
        'clf': (MultiOutputClassifier(RandomForestClassifier()),),
        'clf__estimator__n_estimators':[50]
    }, {
        'vect__max_df':[0.7, 1.0],
        'tfidf__use_idf': [True, False],  
        'clf': (MultiOutputClassifier(LinearSVC()),),
        'clf__estimator__C': [0.5, 0.75, 1.0],
        'clf__estimator__dual': [False],
        'clf__estimator__penalty': ['l1', 'l2']
    }
    
    cv = GridSearchCV(pipeline, parameters, n_jobs=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model based on the precision, recall and f1 scores.
    Prints the average f1 score for the model and the best parameters
    according to GridSearchCV.

    Args:
        model: the model created by the build_model() function
        X_test: the test set with the independent X variables
        Y_test: the test set with the dependent Y variables
        category_names: list with the names of each category

    Returns:
        scores: 2d list with precision, recall and f1 scores 
            for each category
    '''

    Y_pred = model.predict(X_test)

    scores = []
    for i in range(len(category_names)):
        scores.append([ 
            f1_score(Y_test.iloc[:,i], Y_pred[:,i], average='micro'),
            precision_score(Y_test.iloc[:,i], Y_pred[:,i], average='micro'),
            recall_score(Y_test.iloc[:,i], Y_pred[:,i], average='micro')])

    scores = pd.DataFrame(scores, columns=['f1 score', 'precision', 'recall'], 
                        index=category_names)

    print('Average F1 score: ', round(scores.mean(axis=0)[0], 3))
    print('Best parameters: ', model.best_params_, '\n\n')
    

    return scores

def save_model(model, model_filepath):
    '''
    Saves the model to a pickle file
    '''

    joblib.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        

        print('Training model...')
        startTime = datetime.now()
        model.fit(X_train, Y_train)
        print('Training time: ', datetime.now() - startTime)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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