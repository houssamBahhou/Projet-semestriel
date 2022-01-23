import plotly.express as px
import dash_bootstrap_components as dbc
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
from dash import no_update
import dash_table
import pandas as pd
from app import app

import base64
import datetime
import io
from uploading import uploading_layout, overview_layout

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import regex as re
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.linear_model import Ridge
################

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


def processUsagesApps(df):
       apps = df['Apps'].map(lambda x: x.split(', '))
       usages = df['Usages'].map(lambda x: [y if y in classes else 'Autre Usage' for y in pattern.split(x)])
       encoded1 = pd.DataFrame(mlb.fit_transform(apps), columns=mlb.classes_, index=df.index)
       encoded2 = pd.DataFrame(mlb2.fit_transform(usages), columns=mlb2.classes_, index=df.index)
       df[encoded1.columns] = encoded1
       df[encoded2.columns] = encoded2
       df.drop(['Usages','Apps'], axis=1, inplace=True)
       return df
       
def processNewUsagesApps(df):
       apps = df['Apps'].map(lambda x: x.split(', '))
       usages = df['Usages'].map(lambda x: [y if y in classes else 'Autre Usage' for y in pattern.split(x)])
       encoded1 = pd.DataFrame(mlb.transform(apps), columns=mlb.classes_, index=df.index)
       encoded2 = pd.DataFrame(mlb2.transform(usages), columns=mlb2.classes_, index=df.index)
       df[encoded1.columns] = encoded1
       df[encoded2.columns] = encoded2
       df.drop(['Usages','Apps'], axis=1,inplace=True)
       return df



def fullNewDataProcessing(df):
    df_prepared = Itinieraire_frequence_pipeline.transform(df)
    df[['Itinéraire','Fréquence']] = pd.DataFrame(df_prepared, columns=['Itinéraire','Fréquencé'], index=df.index)
    df = processNewUsagesApps(df)
    df_array = num_pipeline.transform(df)
    df = pd.DataFrame(df_array, columns=df.columns, index=df.index)
    return df

Itinieraire_frequence_pipeline = None
num_pipeline = None
mlb = MultiLabelBinarizer()
mlb2 = MultiLabelBinarizer()



def process(df):
    df=pd.DataFrame(df)
    print(df)
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
    dbc.Button(
            "Process", id="example-button", className="me-2", n_clicks=0
        ),
    html.Hr(),
    html.H5("Processed Data"),
    html.Div(id="div-1"),
    # dash_table.DataTable(
    #         id = "example-output",
    #         # editable=True,              # allow editing of data inside all cells
    #         # filter_action="native",     # allow filtering of data by user ('native') or not ('none')
    #         # sort_action="native",       # enables data to be sorted per-column by user or not ('none')
    #         # sort_mode="single",         # sort across 'multi' or 'single' columns
    #         # column_selectable="multi",  # allow users to select 'multi' or 'single' columns
    #         # row_selectable="multi",     # allow users to select 'multi' or 'single' rows
    #         # row_deletable=True,         # choose if user can delete a row (True) or not (False)
    #         # selected_columns=[],        # ids of columns that user selects
    #         # selected_rows=[],           # indices of rows that user selects
    #         # page_action="native",       # all data is passed to the table up-front or not ('none')
    #         # page_current=0,             # page number that user is on
    #         page_size=4,                # number of rows visible per page
    #         style_cell={
    #             'textAlign': 'left',
    #             'whiteSpace': 'normal',
    #             'height': 'auto',
    #         },
        #),
])

@app.callback(
    Output('div-1', 'children'),
    [Input("example-button", "n_clicks"),
    State('raw-data', 'data')    ]
)
def on_button_click(n,data):
    if n==0:
        return []
    else:
        d = process(data)
        return [
                dash_table.DataTable(
                    data=d.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in d.columns],
                    page_size=4,                # number of rows visible per page
                    style_cell={
                        'textAlign': 'left',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    #row_selectable=True,
                    #selected_row_indices=list(data.index),  # all rows selected by default
                )]




## Plan for processing

