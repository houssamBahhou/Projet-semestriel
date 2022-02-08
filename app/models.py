import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
import pandas as pd
from app import app
import plotly.graph_objects as go
import dash_daq as daq
import plotly.express as px
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

models = ['RF', 'KNN', 'GNB', 'DT']
FONTSIZE = 15
FONTCOLOR = "#F5FFFA"
BGCOLOR ="#3445DB"

models_layout = html.Div([
    html.P("In this Dashboard you can train several Machine Learning Models and try to predict NPS, and see which features influence the most in this prediction"),
    html.Hr(),
    html.Div(id="div-2"),
   
])

@app.callback(
    Output('div-2', 'children'),
    [Input("processed-data", "data")]
)
def shoow_dashboard(data):
    dff= pd.DataFrame(data)
    y = dff[['NPS']]
    X = dff.drop('NPS',axis=1)
    return dbc.Row([
        dbc.Col(
                [
                    html.Div(id='test-split'),
                    dcc.Slider(
                        id='slider-split',
                        min=0, max=100, value=10,
                        marks={i: str(i) for i in range(0,110,10)}),
                    html.P("Select Variables", className="control_label"),
                    dcc.Dropdown(
                        id="select_independent",
                        options=[{'label':x, 'value':x} for x in X.columns],
                        value= list(X.columns),
                        multi=True,
                        className="dcc_control",
                    ),
                    html.Div(
                        [
                        html.Div(
                            [
                                html.P("Select Number of KFOLD Splits", className="control_label"),
                                dbc.Row([
                                dbc.Col(
                                daq.NumericInput(
                                    id='id-daq-splits',
                                    min=0,
                                    max=5,
                                    size = 100,
                                    value=2
                                )),
                                dbc.Col([
                                dbc.Button(
                                    "About KFOLD", id="popover-bottom-target", color="info"
                                ),
                                dbc.Popover(
                                    [
                                        dbc.PopoverBody(
                                            """KFOLD-Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data.
                                            K refers to the number of groups that a given data sample is to be split into"""),
                                    ],
                                    id="popover",
                                    target="popover-bottom-target",  # needs to be the same as dbc.Button id
                                    placement="bottom",
                                    is_open=False,
                                ), ]) 

                                ])
                            ])]),
                    html.P("Models", className="control_label"),
                    dcc.Dropdown(
                        id="select_models",
                        options = [{'label':x, 'value':x} for x in models],
                        value = models,
                        multi=True,
                        clearable=False,
                        className="dcc_control",
                    ),
                    html.Div(
                        id = 'best-model', style={'color': 'blue', 'fontSize': 15} 
                    ),
                    html.Br(),
                    daq.PowerButton(
                        id = 'id-daq-switch-model',
                        on='True',
                        color='#1ABC9C', 
                        size = 75,
                        label = 'Initiate Model Buidling'
                    ) ,
                    html.H4('Best Model Stats'),
                    dbc.Row(
                    [
                        dbc.Col(
                        daq.LEDDisplay(
                            id='trainset',
                            #label="Default",
                            value=0,
                            label = "Train",
                            size=FONTSIZE,
                            color = FONTCOLOR,
                            backgroundColor=BGCOLOR
                        )),
                        dbc.Col(
                        daq.LEDDisplay(
                            id='testset',
                            #label="Default",
                            value=0,
                            label = "Test",
                            size=FONTSIZE,
                            color = FONTCOLOR,
                            backgroundColor=BGCOLOR
                        )),
                        dbc.Col(
                        daq.LEDDisplay(
                            id='accuracy',
                            #label="Default",
                            value=0,
                            label = "Accuracy",
                            size=FONTSIZE,
                            color = FONTCOLOR,
                            backgroundColor=BGCOLOR
                        )),   
 
                    ]),
                ],           
                className="mb-4",
            ),
        dbc.Col([
            
            dbc.Row(
            [
            html.H4("Chose the NPS class"),
            dbc.Row([
            dbc.Col(
            dcc.Dropdown(id='chose_nps', options=[{'label':x, 'value':x} for x in y['NPS'].unique()],
                value=3, clearable=False, style={"color": "#000000"})),
            dbc.Col([
                dbc.Button(
                    "About ROC-AUC", id="popover-button", color="info"
                ),
                dbc.Popover(
                    [
                        dbc.PopoverBody(
                            """ROC (Receiver Operating Characteristics).
                             AUC (Area Under The Curve).  
                            AUC - ROC curve is a performance measurement for the classification problems at various threshold settings.
                            ROC is a probability curve and AUC represents the degree or measure of separability. 
                            It tells how much the model is capable of distinguishing between classes. Higher the AUC, the better the model is at predicting 0 classes as 0 and 1 classes as 1. By analogy, the Higher the AUC, the better the model is at distinguishing between patients with the disease and no disease.\n
                            for more information visit this website: https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5"""),
                    ],
                    id="popover-auc",
                    target="popover-button",  
                    placement="bottom",
                    is_open=False,
                )])]),
            html.Div(id='fig_roc')      
            ]),
            dbc.Row(

                [dcc.Graph(id="aggregate_graph1", figure = {})],

            ),
               
        ]),
        
    ])


@app.callback(
    [
        Output('test-split','children'),
        Output("accuracy", 'value'),
        Output("trainset", 'value'),
        Output("testset", 'value'),
        Output("best-model", 'children'),
        Output('aggregate_graph1','figure')
        
        
    ],
    [   Input("processed-data", "data"),
        Input("select_independent", "value"),
        Input("id-daq-splits", "value"),
        Input("slider-split", "value"),
        Input("select_models", "value"),
        Input("id-daq-switch-model", 'on')     
    ]
)
def measurePerformance(data, independent, kfolds, slider, selected_models,on):
    if not on:
        dff= pd.DataFrame(data)
        y = dff[['NPS']]
        X = dff.drop('NPS',axis=1)
        accuracy, trainX, testX, bestModel = getModels(X,y,independent,kfolds, slider, selected_models)
        ft_import = featureImportance(X[independent],y)
        return  ['Train / Test split size: {} / {}'.format(slider, 100-slider), round(accuracy,2), trainX, testX, f'The best performing model is {bestModel} with accuracy of {round(accuracy,2)}. Try for various K FOLD values to explore further.', ft_import]



def getModels(X,y, independent, kfolds, slider, selected_models):   
    X=X[independent]     
    n_models = len(selected_models)
    cv_scores = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size = slider/100, random_state = 12)
    trained_models = {}
    # instantiate the model 
    model = GaussianNB()
    modelName = 'RF'
    for i in range(n_models):
        if selected_models[i] == 'GNB':
            model = GaussianNB()
            modelName = 'GNB'
        elif selected_models[i] == 'KNN':
            model = KNeighborsClassifier()
            modelName = 'KNN'
        elif selected_models[i] == 'RF':
            model = RandomForestClassifier()
            modelName = 'RF'
        elif selected_models[i] == 'DT':
            model = DecisionTreeClassifier()
            modelName = 'DT'
        # fit the model 
        model.fit(X_train, y_train)
        trained_models[modelName] = model
        cv_scores[selected_models[i]] = cross_val_score(model, X_train, y_train,scoring="accuracy", cv=kfolds).mean()
        bestModel = max(cv_scores, key=cv_scores.get)
    return max(cv_scores.values()), X_train.shape[0], X_test.shape[0], bestModel


  
def featureImportance(X,y):
    rf = RandomForestClassifier()
    rf.fit(X, y.values.ravel())
    
    sorted_idx = rf.feature_importances_.argsort()
    fig_featureImp = px.bar(y=X.columns[sorted_idx], x=rf.feature_importances_[sorted_idx], labels={
                     "x": "Importance",
                     "y": "Features"},title= 'Variable Importance')
    return fig_featureImp


@app.callback(
    [   
        Output('fig_roc','children')
    ],
    [   
        Input("processed-data", "data"),
        Input("select_independent", "value"),
        Input("slider-split", "value"),
        Input("select_models", "value"),
        Input("chose_nps","value"),
        Input("id-daq-switch-model", 'on')     
    ]
)
def RocAuc(data,independent,slider,selected_models,nps,on):
    if not on:
        dff= pd.DataFrame(data)
        y = dff[['NPS']]
        X = dff.drop('NPS',axis=1)
        X=X[independent]     
        n_models = len(selected_models)
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size = slider/100, random_state = 12)
        trained_models = {}
        # instantiate the model 
        model = GaussianNB()
        modelName = 'RF'
        for i in range(n_models):
            if selected_models[i] == 'GNB':
                model = GaussianNB()
                modelName = 'GNB'
            elif selected_models[i] == 'KNN':
                model = KNeighborsClassifier()
                modelName = 'KNN'
            elif selected_models[i] == 'RF':
                model = RandomForestClassifier()
                modelName = 'RF'
            elif selected_models[i] == 'DT':
                model = DecisionTreeClassifier()
                modelName = 'DT'
            # fit the model 
            model.fit(X_train, y_train)
            trained_models[modelName] = model
        fpr = {}
        tpr = {}
        roc_auc = {}
        predicted_proba ={}
        for i in trained_models:
            predicted_proba[i] = trained_models[i].predict_proba(X)
        fig = go.Figure()
        fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
        )
        for i in predicted_proba: 
            fpr[i], tpr[i], thresholds = roc_curve(y['NPS'],predicted_proba[i][:,nps],pos_label=6)
            roc_auc[i] = auc(fpr[i], tpr[i])    
            name = f"{i} (ROC-AUC={round(roc_auc[i],2):.4f})"
            fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], name=name, mode='lines'))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500,
            title="Receiver Operating Characteristic (ROC) Curve - " + 'Classe: ' + str(nps)
        )
    
        return [dcc.Graph(figure=fig)]

@app.callback(
    Output("popover", "is_open"),
    [Input("popover-bottom-target", "n_clicks")],
    [State("popover", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("popover-auc", "is_open"),
    [Input("popover-button", "n_clicks")],
    [State("popover-auc", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open