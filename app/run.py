import json

import joblib
import pandas as pd
import plotly
from flask import Flask
from flask import render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('disaster', engine)
# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create dictionary for mapping of the 'related' column
    related_dict = {0: 'not_related', 1: 'related'}

    # extract 'related' column and set x & y values
    related = df['related'].map(related_dict)
    related_counts = related.value_counts()
    related_names = list(related_counts.index)

    # extract the the rest of the categories excluding 'related' and set x &
    # y values
    categories = df.iloc[:, 4:]
    categories_mean = categories.mean().sort_values(ascending=False)[1:11]
    categories_names = list(categories_mean.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                plotly.graph_objs.Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Message Genres Distribution',
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
                plotly.graph_objs.Bar(
                    x=related_names,
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'Message Relevance Distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
        {
            'data': [
                plotly.graph_objs.Bar(
                    x=categories_names,
                    y=categories_mean
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories Proportions',
                'yaxis': {
                    'title': "Proportion"
                },
                'xaxis': {
                    'title': ""
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    json_graph = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=json_graph)


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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()