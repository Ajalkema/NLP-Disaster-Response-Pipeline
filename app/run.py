import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import joblib

# import the load and clean data functions
import sys
sys.path.insert(1, './data')
from process_data import load_data, clean_data

app = Flask(__name__)

# load and clean the data
df = load_data('data/disaster_messages.csv', 'data/disaster_categories.csv')
df = clean_data(df)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays visuals and receives user input text for the model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # graph 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # graph 2
    social_messages = ' '.join(df[df['genre'] == 'social']['message'])
    social_messages_tok = tokenize(social_messages)
    social_messages_cnt = pd.Series(social_messages_tok).value_counts()
    social_messages_df = social_messages_cnt.rename_axis('unique_values').reset_index(name='counts')


    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=social_messages_df['unique_values'].iloc[:30],
                    y=social_messages_df['counts'].iloc[:30]
                 )
            ],

            'layout': {
                'title': 'Most used keywords in the "social" genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message"
                }
            }
        }        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=8080, debug=True)


if __name__ == '__main__':
    main()