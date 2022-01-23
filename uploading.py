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
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df = []

data_columns = [
    "Horodateur",
    "Quel âge avez-vous ?",
    "De quel département êtes-vous ?",
    "Comment décririez-vous votre usage des transports routiers en NA ?",
    "Quels sont vos usages de transport en commun",
    "À quelle fréquence utilisez-vous les transports routiers de votre région ?",
    "Quelle application utilisez-vous pour réaliser un itinéraire de transport ?",
    "Saviez-vous que vous pouviez réaliser un itinéraire depuis le portail de la Nouvelle Aquitaine ?",
    " Aller sur le site transport de la NA, puis indiquez quelle est votre niveau appréciation globale du portail ?",
    "En allant sur le portail de la nouvelle Aquitaine, quelles sont les informations que vous souhaiteriez trouver ?",
    "Que pensez-vous de l’esthétisme du site ?",
    "Pourriez-vous simuler un itinéraire,  puis évaluer votre niveau de satisfaction du service",
    "Qu’est-ce que vous avez apprécié ?",
    "Qu’est-ce que vous avez le plus aimé dans le design du site ?",
    "Qu’est-ce que tu avez le moins aimé dans le design du site ?",
    "Comment pourrions-nous améliorer le portail d'un point de vue accessibilité ?",
    "sus items",
    "1","2","3","4","5","6","7","8","9","10",
    "Quelle est la probabilité que vous recommandiez le portail de la Nouvelle Aquitaine à un ami ou un collègue ?",
    "Souhaitez-vous partager un commentaire, remarque ou suggestion ?",
]

uploading_layout= html.Div([
    dbc.Row([
        dbc.Col([ 
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            )],
            width=12,
        )
    ],
    className="mt-2"
    ),
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            if len(df.columns)!=len(data_columns):
                return html.Div([
                    dcc.ConfirmDialog(
                            message="The Data uploaded columns don't match the expected columns",
                            displayed=True,
                        ), 
                ])           
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        html.Hr(),
        # html.H5("Raw Data"),
        # dash_table.DataTable(
        #     data=df.to_dict('records'),
        #     columns=[{'name': i, 'id': i} for i in df.columns],
        #     # editable=True,              # allow editing of data inside all cells
        #     # filter_action="native",     # allow filtering of data by user ('native') or not ('none')
        #     # sort_action="native",       # enables data to be sorted per-column by user or not ('none')
        #     # sort_mode="single",         # sort across 'multi' or 'single' columns
        #     # column_selectable="multi",  # allow users to select 'multi' or 'single' columns
        #     # row_selectable="multi",     # allow users to select 'multi' or 'single' rows
        #     # row_deletable=True,         # choose if user can delete a row (True) or not (False)
        #     # selected_columns=[],        # ids of columns that user selects
        #     # selected_rows=[],           # indices of rows that user selects
        #     # page_action="native",       # all data is passed to the table up-front or not ('none')
        #     # page_current=0,             # page number that user is on
        #     page_size=4,                # number of rows visible per page
        #     style_cell={
        #         'textAlign': 'left',
        #         'whiteSpace': 'normal',
        #         'height': 'auto',
        #     },
        # ),
        #html.Hr(),
        html.H5("Data information"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                    [
                        dbc.CardBody([
                            html.H4("Number of Instances", className="card-title"),
                            html.P(df.shape[0], style={"text-align":"center","font-size":"100px","margin-top":"7px", "color":"white"}),
                        
                        ]),
                    ], style={"width": "25rem","height":"220px","border-radius": "10px"},color="info",inverse=True)),
                #width=4),
                dbc.Col(
                    dbc.Card(
                    [
                        dbc.CardBody([
                            html.H4("Number of Features", className="card-title"),
                            html.P(df.shape[1],style={"text-align":"center","font-size":"100px","margin-top":"7px", "color":"white"}),
                        
                        ]),
                    ], style={"width": "25rem","height":"220px","border-radius": "10px"},color="info",inverse=True)),
                dbc.Col(
                    dbc.Card(
                    [
                        dbc.CardBody([
                            html.H5("Statistic of each Feature", className="card-title"),
                            dcc.Dropdown(id='ft_choice', options=[{'label':x, 'value':x} for x in df.columns],
                             value='Horodateur', clearable=False, style={"color": "#000000"}),
                            html.Div(id='info_ft')
                        ]),
                    ],style={"border-radius": "10px"},color="info",inverse=True)),
                
                #width=4),
                #dbc.Col(second_card, width=8),
            ],
            
            className="mb-4",
        ),
        dbc.Row([
                dbc.Col(
                    dbc.Card(
                    
                        dcc.Graph(id='null_values', figure={}), body=True, color="secondary",)
                )
            ],
             className="mb-4",
            ),
        dcc.Store(id='stored-data', data=df.to_dict('records')),
        html.Hr(),  # horizontal line
        html.P("Inset X axis data"),
        dcc.Dropdown(id='xaxis-data',
                     options=[{'label':x, 'value':x} for x in df.columns]),
        html.P("Inset Y axis data"),
        dcc.Dropdown(id='yaxis-data',
                     options=[{'label':x, 'value':x} for x in df.columns]),
        html.Button(id="submit-button", children="Create Graph"),
        
    ])
overview_layout =  html.Div([   
    
    html.Div(id='output-data'),
    html.Div(id='output-div'),
    ])   
   

@app.callback(Output('raw-data', 'data'),
              Input('stored-data', 'data'))
def copy_raw_data(d):
    return d

@app.callback(Output('output-data', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(Output('output-div', 'children'),
              Input('submit-button','n_clicks'),
              State('stored-data','data'),
              State('xaxis-data','value'),
              State('yaxis-data', 'value'))
def make_graphs(n, data, x_data, y_data):
    if n is None:
        return no_update
    else:
        bar_fig = px.bar(data, x=x_data, y=y_data)
        return dcc.Graph(figure=bar_fig)

@app.callback(Output('info_ft','children'),
                Input('ft_choice','value'),
                State('stored-data','data'))
def give_ft_info(ft,data):
    dff= pd.DataFrame(data)
    print("hada howaa",dff[ft].dtype.name)
    columnss= []
    if dff[ft].dtype.name == 'object':
        columnss=['count','unique','freq']
        dff=dff[ft].describe().reset_index()
    elif dff[ft].dtype.name == 'int64':
        dff=dff[ft].describe().apply("{0:.2f}".format).reset_index()
        columnss=['count','mean','min','max']
    
    dff.columns = ["metric","value"]
    dff.set_index("metric",inplace=True)
    print(dff.loc[columnss])
    return dash_table.DataTable(
            
            data=dff.loc[columnss].reset_index().to_dict("records"),
            columns=[{'name': i, 'id': i,} for i in ["metric","value"]],
            page_size=8,                # number of rows visible per page
            style_cell={
                'backgroundColor': '#828587',
                'textAlign': 'left',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
        ),

@app.callback(
    Output("null_values", "figure"),
    [Input("stored-data", "data")]
)
def plot_graph_null(data):
    dff= pd.DataFrame(data)
    fig = px.bar(dff.isnull().sum(), labels={
                     "index": "Columns",
                     "value": "Count of NaN values"}, title="Number of NaN values in the dataset")
    return fig
