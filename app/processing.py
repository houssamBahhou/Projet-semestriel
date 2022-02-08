import plotly.express as px
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
import dash_table
import pandas as pd
from app import app
import plotly.graph_objects as go
from sklearn.decomposition import PCA

import numpy as np
import regex as re
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



dict = {'Quel âge avez-vous ?': 'Age',
        'Quels sont vos usages de transport en commun': 'Usages',
        'De quel département êtes-vous ?': 'Dépratement',
        'À quelle fréquence utilisez-vous les transports routiers de votre région ?': 'Fréquence',
        'Quelle application utilisez-vous pour réaliser un itinéraire de transport ?':'Apps',
        'Saviez-vous que vous pouviez réaliser un itinéraire depuis le portail de la Nouvelle Aquitaine ? ': 'Itinéraire',
        ' Aller sur le site transport de la NA, puis indiquez quelle est votre niveau appréciation globale du portail ?': 'Appréciation-portail',
        'Que pensez-vous de l’esthétisme du site ?': 'Esthétisme',
        'Pourriez-vous simuler un itinéraire,  puis évaluer votre niveau de satisfaction du service': 'Itinéraire-satisfaction',
        'Quelle est la probabilité que vous recommandiez le portail de la Nouvelle Aquitaine à un ami ou un collègue ? ': 'NPS',
        '1':'Utilisation-fréquente',
        '2':'systeme-complexe',
        '3':'systeme-facile',
        '4':'support-specialiste',
        '5':'fonctions-intégrées',
        '6':'systeme_incoherent',
        '7':'facile-apprendre',
        '8':'systeme-contraignant',
        '9':'confiance',
        '10':'familiarisation-difficile',
        }



def processAge(ages):
    for i in range(len(ages)):
        ages[i] = int(re.sub(r'(?P<age>[0-9]+).*',r"\g<age>",ages[i]))
    return ages


Itinieraire_frequence_pipeline = None
num_pipeline = None
mlb = MultiLabelBinarizer()
mlb2 = MultiLabelBinarizer()


def process(df):
    df=pd.DataFrame(df)
    data = df.drop(['Horodateur','Comment décririez-vous votre usage des transports routiers en NA ?','En allant sur le portail de la nouvelle Aquitaine, quelles sont les informations que vous souhaiteriez trouver ?','Qu’est-ce que vous avez apprécié ?','Qu’est-ce que vous avez le plus aimé dans le design du site ?','Qu’est-ce que tu avez le moins aimé dans le design du site ?',"Comment pourrions-nous améliorer le portail d'un point de vue accessibilité ?",'sus items','Souhaitez-vous partager un commentaire, remarque ou suggestion ?'],axis=1)    
    data.rename(columns=dict,
            inplace=True)

    #drop rows that dont contain NPS and split data (features + Label)
    data = data.drop(index=[0,1,2,3], axis=0)
    data = data.reset_index()
    data = data.drop(['index'],axis=1)
    processAge(data['Age'])
    #processing text features: Itinéraire and Fréquence

    pipeline = Pipeline([
    ("imp", SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ("cat", OrdinalEncoder(categories=[['Non, je ne le savais pas', 'Oui, je le savais'],['Plusieurs fois par semaine', 'Quelques fois par an',
            'Quelques fois par mois','Jamais']]))
    ])

    Itinieraire_frequence_pipeline = ColumnTransformer([
    ("cat", pipeline, ["Itinéraire","Fréquence"]),
    ])

    data_prepared = Itinieraire_frequence_pipeline.fit_transform(data)
    data[['Itinéraire','Fréquence']] = pd.DataFrame(data_prepared, columns=['Itinéraire','Fréquencé'], index=data.index)

    #processing text features: Usages and Apps

    pattern = re.compile('(?<!sport|courses), ')
    classes = ['Activités de la vie quotidienne (aller au sport, faire les courses, ...)',
        'Déplacement professionnel occasionnel',
        'Déplacement professionnel quotidien', 'Tourisme','Transport scolaire']
    apps = data['Apps'].map(lambda x: x.split(', '))
    usages = data['Usages'].map(lambda x: [y if y in classes else 'Autre Usage' for y in pattern.split(x)])
    encoded1 = pd.DataFrame(mlb.fit_transform(apps), columns=mlb.classes_, index=data.index)
    encoded2 = pd.DataFrame(mlb2.fit_transform(usages), columns=mlb2.classes_, index=data.index)
    data[encoded1.columns] = encoded1
    data[encoded2.columns] = encoded2
    data.rename(columns={"Le portail de la Nouvelle Aquitaine et des transports":"portail",
                        "Activités de la vie quotidienne (aller au sport, faire les courses, ...)":"Activités",
                        "Déplacement professionnel occasionnel":"Déplacement prof_occas",
                        'Déplacement professionnel quotidien':"Déplacement prof_quot"},
            inplace=True)
    data.drop(['Usages','Apps'], axis=1, inplace=True)
    y = data[['NPS']]
    X = data.drop('NPS',axis=1)
       
    #fill missing numerical values 
    num_pipeline = Pipeline([
    ('imput', SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
    ('std_scaler', StandardScaler()),
    ])
    X_array = num_pipeline.fit_transform(X)
    X = pd.DataFrame(X_array, columns=X.columns, index=X.index, dtype=float)
    final_data = pd.concat([X,y], axis=1)   
    return final_data


processing_layout = html.Div([
    html.P("In this section we provide an example of processing of NA data transport"),
    html.Hr(),
    html.Div(id="div-1"),
   
])

@app.callback(
    Output('div-1', 'children'),
    [
    Input('raw-data', 'data')    ]
)
def on_button_click(data):
    d = process(data)
    return [
        dcc.Store(id='stored-processed-data', data=d.to_dict('records')),
        html.H5("Processed Data information"),
        dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                [
                    dbc.CardBody([
                        html.H4("Number of Instances", className="card-title"),
                        html.P(d.shape[0], style={"text-align":"center","font-size":"100px","margin-top":"7px", "color":"white"}),
                    
                    ]),
                ], style={"width": "25rem","height":"220px","border-radius": "10px"},color="info",inverse=True)),
            dbc.Col(
                dbc.Card(
                [
                    dbc.CardBody([
                        html.H4("Number of Features", className="card-title"),
                        html.P(d.shape[1],style={"text-align":"center","font-size":"100px","margin-top":"7px", "color":"white"}),
                    
                    ]),
                ], style={"width": "25rem","height":"220px","border-radius": "10px"},color="info",inverse=True)),
            dbc.Col(
                dbc.Card(
                [
                    dbc.CardBody([
                        html.H5("Statistic of each Feature", className="card-title"),
                        dcc.Dropdown(id='p_ft_choice', options=[{'label':x, 'value':x} for x in d.columns],
                            value='Utilisation-fréquente', clearable=False, style={"color": "#000000"}),
                        html.Div(id='info_p_ft')
                    ]),
                ],style={"border-radius": "10px"},color="info",inverse=True)),

        ],
            className="mb-4",
        ),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Features distribution", className="card-title"),
                        dcc.Dropdown(id='chose_hist', options=[{'label':x, 'value':x} for x in d.columns],
                                    value='Utilisation-fréquente', clearable=False, style={"color": "#000000"}),
                        html.Div(id='ft_hist')
                    ])
                ])
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Features Correlation", className="card-title"),
                        dcc.Graph(figure = go.Figure(data = go.Heatmap(
                                z=d.corr().values,
                                x=d.corr().columns,
                                y=d.corr().columns,
                                colorscale=px.colors.diverging.RdBu,
                                zmin=-1,
                                zmax=1
                            ), layout=go.Layout(autosize=False,width=700,height=500,margin=go.layout.Margin(
                                                                                                            l=20,
                                                                                                            r=20,
                                                                                                            b=20,
                                                                                                            t=20,
                                                                                                            pad = 2
                                                                                                        ))))
                    ])
                ])
            )
        ], 
            className="mb-4",
        ),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            dcc.Graph(id="pca_multi"),
                            html.P("Number of components:"),
                            dcc.Slider(
                                id='slider',
                                min=2, max=5, value=3,
                                marks={i: str(i) for i in range(2,6)}),
                            
                        ]),
                        dash_table.DataTable(
                            id="important_ft",
                            page_size=8,                # number of rows visible per page
                            style_cell={
                                'backgroundColor': '#02b9b7',
                                'textAlign': 'left',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'color':'white',
                                'font-size':'20px'
                            },
                        ),
                    ])
                ])
            )
        ])    
    ]
                


@app.callback(Output('info_p_ft','children'),
                Input('p_ft_choice','value'),
                State('stored-processed-data','data'))
def give_p_ft_info(ft,data):
    dff= pd.DataFrame(data)
    columnss= ['count','mean','min','max']
    dff=dff[ft].describe().apply("{0:.2f}".format).reset_index()
    dff.columns = ["metric","value"]
    dff.set_index("metric",inplace=True)
    return dash_table.DataTable(     
            data=dff.loc[columnss].reset_index().to_dict("records"),
            columns=[{'name': i, 'id': i,} for i in ["metric","value"]],
            page_size=8,                # number of rows visible per page
            style_cell={
                'backgroundColor': '#828587',
                'textAlign': 'left',
                'whiteSpace': 'normal',
                'height': 'auto',
                'width':'60%'
            },
        ),


@app.callback(
    Output("ft_hist", "children"),
    [Input("chose_hist","value"),
    State("stored-processed-data", "data")]
)
def plot_hist(ft,data):
    dff= pd.DataFrame(data)
    fig = px.histogram(dff[ft],nbins=20,labels={
                     "index": ""},title=ft)
    return dcc.Graph(figure=fig.update_layout(showlegend = False))


@app.callback(
    Output("pca_multi", "figure"),
    Output("important_ft","data"), 
    Output("important_ft","columns"),
    [Input("slider", "value"),
    State("stored-processed-data", "data")])
def run_and_plot(n_components,data):
    dff= pd.DataFrame(data)
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(dff)
    
    if n_components>2:
        n_ft = 3
    else:
        n_ft=2
    var = pca.explained_variance_ratio_.sum() * 100
    most_important = [np.argsort(-np.abs(components[i]))[:n_ft] for i in range(n_components)]
    most_important_names = [['PC{}'.format(i)]+[dff.columns[most_important[i][j]] for j in range(n_ft)] for i in range(n_components)]

    components_names = ['PC{}'.format(i) for i in range(n_components)]
    columnss=['Top {} feature'.format(i) for i in range(1,n_ft+1)]
    columnss= ['Component']+columnss
    df_pca = pd.DataFrame(data=most_important_names,columns=columnss, index=components_names)
    labels = {str(i): f"PC {i+1}" 
              for i in range(n_components)}
    labels['color'] = 'Median Price'

    fig = px.scatter_matrix(
        components,
        color=dff['NPS'],
        dimensions=range(n_components),
        labels=labels,
        title=f'Total Explained Variance: {var:.2f}%')
    fig.update_traces(diagonal_visible=False)
    return fig,df_pca.reset_index().to_dict("records"),[{'name': i, 'id': i,} for i in df_pca.columns]


@app.callback(Output('processed-data', 'data'),
            Input('stored-processed-data', 'data'))
def copy_processed_data(d):
    return d