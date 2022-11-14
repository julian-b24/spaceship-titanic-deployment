import dash
from dash import dcc
from dash import html
from dash import dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# Model development data exploration/analysis
df = pd.read_csv('train.csv')
df_copy = deepcopy(df)
df_copy = df_copy.dropna()
df_mod = deepcopy(df)

def replace_categories(data, column):
    categories = list(data[column].value_counts().index)
    new_categories = list(range(1, len(categories) + 1))
    data[column].replace(categories, new_categories, inplace=True)

categories = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination']

for category in categories:
    replace_categories(df_mod, category)

df_mod['VIP'].replace([True, False], [1, 0], inplace=True)
df_mod.drop('Name', axis=1, inplace=True)
df_mod['Transported'].replace([True, False], [1, 0], inplace=True)
df_mod.drop('PassengerId', axis=1, inplace=True)

imputer = KNNImputer(n_neighbors=3)
values = imputer.fit_transform(df_mod.values)

df_final = pd.DataFrame(data=values, columns=df_mod.columns)
df_final['PassengerId'] = df['PassengerId']

#Graph
fig = px.bar(df, x="HomePlanet", color="Transported", barmode="group")

#Data partition
target = 'Transported'
attributes = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa']

X = df_final[attributes]
y = df_final[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, 
                                                    shuffle=True)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

app.layout = html.Div(children=[

    

    html.H1(children='Spaceship Titanic'),

    html.Div(children='''
        A web application for this data challenge.
    '''),

    html.Div(children=[

    html.H1(children='Data'),
    html.P(children='Final dataset used for predictions'),

    dash_table.DataTable(df_final.to_dict('records'), 
    [{"name": i, "id": i} for i in df_final.columns],
    page_size=10)

    ]),

    html.Div(children=[

    html.H1(children='EDA'),
    html.P(children='Bar graph as an exploration'),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )

    ]),

    html.Div(children=[

    html.H1(children='Inference'),
    html.P(children='Inference using random forest'),

    html.Img(src='https://storage.googleapis.com/kaggle-media/competitions/Spaceship%20Titanic/joel-filipe-QwoNAhbmLLo-unsplash.jpg',
    width=400, height=320),

    html.Div(children=[

    html.Label('Room Service'),
    dcc.Input(id ='room-service', value='0', type='number', style={'margin':'10px'}),

    html.Label('Food Court'),
    dcc.Input(id ='food-court', value='0', type='number', style={'margin':'10px'}),

    html.Label('Shopping Mall'),
    dcc.Input(id ='shopping-mall', value='0', type='number', style={'margin':'10px'}),

    html.Label('Spa'),
    dcc.Input(id ='spa', value='0', type='number', style={'margin':'10px'}),
    ]),
    
    ]),

    html.Div(id="result")


])


@app.callback(
    Output("result", "children"),
    Input("room-service", "value"),
    Input("food-court", "value"),
    Input("shopping-mall", "value"),
    Input("spa", "value"),
)
def update_output(input1, input2, input3, input4):
    data= np.array([input1, input2, input3, input4])
    entry =pd.DataFrame(data)
    test_pred = rf_model.predict(entry)
    
    return u'This person was transported:  {}'.format(test_pred)



if __name__ == '__main__':
    app.run_server(debug=True)