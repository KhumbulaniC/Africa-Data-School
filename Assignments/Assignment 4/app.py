# Dash_App.py
# Import Packages
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import pickle
# Setup
app = dash.Dash(__name__)
app.title = 'Machine Learning Model Deployment'
server = app.server
# load ML model
with open('churn_model.pickle', 'rb') as f:
    clf = pickle.load(f)
# App Layout
app.layout = html.Div([
    dbc.Row([html.H3(children='Predict Banking Customer Churn')]),
    dbc.Row([
        dbc.Col(html.Label(children='Credit Score:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='CreditScore', type='text', value=619)) 
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Age:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='Age', type='text', value=42)) 
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Tenure:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='Tenure', type='text', value=2))  
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Balance:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='Balance', type='text', value=0))  
    ]),  
    dbc.Row([
        dbc.Col(html.Label(children='Number of Products:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='NumOfProducts', type='text', value=1)) 
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Has Credit Card:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='HasCrCard', type='text', value=1)) 
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Is Active Member:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='IsActiveMember', type='text', value=1))  
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Estimated Salary:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='EstimatedSalary', type='text', value=101348.88))  
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Geography_Germany:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='Geography_Germany', type='text', value=0))  
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Geography_Spain:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='Geography_Spain', type='text', value=0))  
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Gender is Male:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='Gender_Male', type='text', value=0))  
    ]),
    dbc.Row([dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary")]),
    html.Br(),
    dbc.Row([html.Div(id='prediction output')])
    ], style = {'padding': '0px 0px 0px 150px', 'width': '50%'})
# Callback to produce the prediction 
@app.callback(
    Output('prediction output', 'children'),
    Input('submit-val', 'n_clicks'),
    State('CreditScore', 'value'),
    State('Age', 'value'),
    State('Tenure', 'value'), 
    State('Balance', 'value'),
    State('NumOfProducts', 'value'),
    State('HasCrCard', 'value'),
    State('IsActiveMember', 'value'), 
    State('EstimatedSalary', 'value'),
    State('Geography_Germany', 'value'),
    State('Geography_Spain', 'value'),
    State('Gender_Male', 'value')
)
def update_output(n_clicks, CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_Germany, Geography_Spain, Gender_Male):
    x = np.array([[float(CreditScore), float(Age), float(Tenure), float(Balance), float(NumOfProducts), float(HasCrCard), float(IsActiveMember), float(EstimatedSalary), float(Geography_Germany), float(Geography_Spain), float(Gender_Male)]])
    df=pd.DataFrame(data=x[0:,0:], index=[i for i in range(x.shape[0])], columns=[str(i) for i in ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'Gender_Male']])
    prediction = clf.predict(df)
    if prediction == 1:
        output = 'churn'
    elif prediction == 0:
        output = 'remain'
    return f'The prediction is that the customer is like to {output}.'
# Run the App 
if __name__ == '__main__':
    app.run_server()