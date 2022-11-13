import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[

    html.Hr(style={
        'background-color':'black',
        'height': '2px'
    }),

    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    html.Hr(),

    html.Img(src='https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Logo_de_la_Universidad_ICESI.svg/1280px-Logo_de_la_Universidad_ICESI.svg.png',
    width=200, height=75)
])

if __name__ == '__main__':
    app.run_server(debug=True)