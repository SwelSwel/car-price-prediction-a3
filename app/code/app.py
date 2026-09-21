from dash import Dash, dcc, html, Input, Output, State
import joblib
import numpy as np
import pandas as pd
from model_classes import *

# Load A1 model (Random Forest)
old_model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
fill_values = joblib.load('fill_values.pkl')

# Load A2 model (Custom Linear Regression)
new_model = joblib.load('new_model.pkl')

num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
cat_cols = ['brand', 'fuel', 'seller_type', 'transmission']
remaining_cols = ['owner', 'seats']

app = Dash(__name__)

# Shared input fields
def create_input_fields(suffix):
    return [
        html.Label("Year"),
        dcc.Input(id=f'year-{suffix}', type='number'),
        html.Label("Brand"),
        dcc.Dropdown(id=f'brand-{suffix}',
                     options=[{'label': f, 'value': f} for f in encoder.categories_[0]]),
        html.Label("Fuel"),
        dcc.Dropdown(id=f'fuel-{suffix}',
                     options=[{'label': f, 'value': f} for f in encoder.categories_[1]]),
        html.Label("KM Driven"),
        dcc.Input(id=f'km_driven-{suffix}', type='number'),
        html.Label("Seller Type"),
        dcc.Dropdown(id=f'seller_type-{suffix}',
                     options=[{'label': f, 'value': f} for f in encoder.categories_[2]]),
        html.Label("Transmission Type"),
        dcc.Dropdown(id=f'transmission-{suffix}',
                     options=[{'label': f, 'value': f} for f in encoder.categories_[3]]),
        html.Label("Owner"),
        dcc.Input(id=f'owner-{suffix}', type='number'),
        html.Label("Mileage"),
        dcc.Input(id=f'mileage-{suffix}', type='number'),
        html.Label("Engine"),
        dcc.Input(id=f'engine-{suffix}', type='number'),
        html.Label("Max Power"),
        dcc.Input(id=f'max_power-{suffix}', type='number'),
        html.Label("Seats"),
        dcc.Input(id=f'seats-{suffix}', type='number'),
    ]

# Shared preprocessing function
def preprocess(year, brand, fuel, km_driven, transmission,
               owner, mileage, engine, max_power, seats, seller_type):
    sample = pd.DataFrame([{
        'year': year, 'km_driven': km_driven, 'mileage': mileage,
        'engine': engine, 'max_power': max_power, 'owner': owner,
        'seats': seats, 'brand': brand, 'fuel': fuel,
        'seller_type': seller_type, 'transmission': transmission
    }])
    for col in fill_values:
        sample[col] = sample[col].fillna(fill_values[col])
    sample_num = scaler.transform(sample[num_cols])
    sample_cat = encoder.transform(sample[cat_cols])
    sample_pass = sample[remaining_cols].values
    return np.hstack([sample_num, sample_pass, sample_cat])

app.layout = html.Div([
    html.H1("Car Price Predictor"),
    dcc.Tabs([
        dcc.Tab(label='Old Model (Random Forest)', children=[
            html.Div([
                html.H3("Random Forest Model (Assignment 1)"),
                html.P("This is the original Random Forest model trained from Assignment 1. "
                       ),
                html.P("How to use: Fill in all the fields below with your car's details, "
                       "then click Predict to see the estimated selling price."),
                *create_input_fields('old'),
                html.Button("Predict", id='predict_btn_old', n_clicks=0),
                html.H3(id='output-old')
            ])
        ]),
        dcc.Tab(label='New Model (Linear Regression)', children=[
            html.Div([
                html.H3("Custom Linear Regression Model (Assignment 2)"),
                html.P("This is a custom Linear Regression model built from scratch using stochastic gradient descent (lr=0.001). "
                       "It demonstrates gradient descent optimization, cross-validation, and regularization techniques "
                       "implemented from the ground up. It achieves R2=0.884 on the test set."),
                html.P(
                       "This model was built entirely from scratch with implementations of Xavier initialization, "
                       "momentum-based gradient descent, and L1/L2 regularization. It provides interpretable feature importance "
                       "through its learned coefficients."),
                html.P("How to use: Fill in all the fields below with your car's details, "
                       "then click Predict to see the estimated selling price."),
                *create_input_fields('new'),
                html.Button("Predict", id='predict_btn_new', n_clicks=0),
                html.H3(id='output-new')
            ])
        ]),
    ])
])

# Old model callback
@app.callback(
    Output('output-old', 'children'),
    Input('predict_btn_old', 'n_clicks'),
    State('year-old', 'value'), State('brand-old', 'value'),
    State('fuel-old', 'value'), State('km_driven-old', 'value'),
    State('transmission-old', 'value'), State('owner-old', 'value'),
    State('mileage-old', 'value'), State('engine-old', 'value'),
    State('max_power-old', 'value'), State('seats-old', 'value'),
    State('seller_type-old', 'value'),
    prevent_initial_call=True
)
def predict_old(n_clicks, year, brand, fuel, km_driven, transmission,
                owner, mileage, engine, max_power, seats, seller_type):
    sample_final = preprocess(year, brand, fuel, km_driven, transmission,
                              owner, mileage, engine, max_power, seats, seller_type)
    predicted = np.exp(old_model.predict(sample_final))
    return f"The predicted selling price is {predicted[0]:,.2f}"

# New model callback
@app.callback(
    Output('output-new', 'children'),
    Input('predict_btn_new', 'n_clicks'),
    State('year-new', 'value'), State('brand-new', 'value'),
    State('fuel-new', 'value'), State('km_driven-new', 'value'),
    State('transmission-new', 'value'), State('owner-new', 'value'),
    State('mileage-new', 'value'), State('engine-new', 'value'),
    State('max_power-new', 'value'), State('seats-new', 'value'),
    State('seller_type-new', 'value'),
    prevent_initial_call=True
)
def predict_new(n_clicks, year, brand, fuel, km_driven, transmission,
                owner, mileage, engine, max_power, seats, seller_type):
    sample_final = preprocess(year, brand, fuel, km_driven, transmission,
                              owner, mileage, engine, max_power, seats, seller_type)
    # Add intercept column for custom model
    intercept = np.ones((sample_final.shape[0], 1))
    sample_with_intercept = np.concatenate((intercept, sample_final), axis=1)
    predicted = np.exp(new_model.predict(sample_with_intercept))
    return f"The predicted selling price is {predicted[0]:,.2f}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)