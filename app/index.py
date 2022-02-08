from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from app import app
from uploading import uploading_layout, overview_layout
from processing import processing_layout
from models import models_layout
import dash_bootstrap_components as dbc


app_tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="Data Overview", tab_id="tab-mentions", labelClassName="text-success font-weight-bold", activeLabelClassName="text-danger"),
                dbc.Tab(label="Processing", tab_id="tab-trends", labelClassName="text-success font-weight-bold", activeLabelClassName="text-danger"),
                dbc.Tab(label="Model", tab_id="tab-other", labelClassName="text-success font-weight-bold", activeLabelClassName="text-danger"),
            ],
            id="tabs",
            active_tab="tab-mentions",
        ),
    ], className="mt-3"
)

app.layout = dbc.Container([ # this code section taken from Dash docs https://dash.plotly.com/dash-core-components/upload
    
    dbc.Row([
        dbc.Col(html.Img(
                            src=app.get_asset_url("logo-onepoint.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-top": "5px",
                            },
                        )),
        dbc.Col(html.H1("NPS prediction",
                            style={"textAlign": "center","margin-top": "5px"}), width=7),
        dbc.Col(dbc.Button("Visit our website", id = "onepoint",color="secondary",style={"margin-top": "5px"},href="https://www.groupeonepoint.com/fr/",active=False))
    
    ]),
    html.Hr(),
    dcc.Store(id='raw-data'),  
    dcc.Store(id='processed-data'), 
    dbc.Row(dbc.Col(uploading_layout, width=12), className="mb-3"),
    dbc.Row(dbc.Col(app_tabs, width=12), className="mb-3"),
    html.Div(id='interface',children=[]),
])


@app.callback(
    Output("interface", "children"),
    Input("tabs", "active_tab")
)
def switch_tab(tab_chosen):
    if tab_chosen == "tab-mentions":
        return [overview_layout]
    elif tab_chosen == "tab-trends":
        return [processing_layout]
    elif tab_chosen == "tab-other":
        return [models_layout]
    return html.P("This shouldn't be displayed for now...")

@app.callback(
    Output("onepoint", "is_open"),
    Input("onepoint", "n_clicks"),
    State("onepoint", "external_link")
)
def toggle_onepoint(n, is_open):
    if n:
        return not is_open
    return is_open
    
app.run_server(port="8000", host="0.0.0.0")