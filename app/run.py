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
    cols = ['related', 'request', 'offer',
            'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
            'security', 'military', 'child_alone', 'water', 'food', 'shelter',
            'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
            'infrastructure_related', 'transport', 'buildings', 'electricity',
            'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
            'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
            'other_weather', 'direct_report']

    df.loc[:, cols] = df.loc[:, cols].apply(pd.to_numeric)
    # genre and aid_related status
    aid_rel1 = df[df['aid_related'] == 1].groupby('genre').count()['message']
    aid_rel0 = df[df['aid_related'] == 0].groupby('genre').count()['message']
    genre_names = list(aid_rel1.index)

    # let's calculate distribution of classes with 1
    class_distr1 = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum() / len(df)

    # sorting values in ascending
    class_distr1 = class_distr1.sort_values(ascending=False)

    # series of values that have 0 in classes
    class_distr0 = (class_distr1 - 1) * -1
    class_name = list(class_distr1.index)
    # create visuals
    graphs = [
        {
            'data': [
                plotly.graph_objs.Bar(
                    x=genre_names,
                    y=aid_rel1,
                    name='Aid related'

                ),
                plotly.graph_objs.Bar(
                    x=genre_names,
                    y=aid_rel0,
                    name='Aid not related'
                )
            ],

            'layout': {
                'title': 'Distribution of messages ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'stack'
            }
        },
        {
            'data': [
                plotly.graph_objs.Bar(
                    x=class_name,
                    y=class_distr1,
                    name='Class = 1'
                    # orientation = 'h'
                ),
                plotly.graph_objs.Bar(
                    x=class_name,
                    y=class_distr0,
                    name='Class = 0',
                    marker=dict(
                        color='rgb(212, 228, 247)'
                    )
                    # orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Distribution of labels',
                'yaxis': {
                    'title': "Distribution"
                },
                'xaxis': {
                    'title': "Class",
                           'tickangle': -45
                },
                'barmode': 'stack'
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
