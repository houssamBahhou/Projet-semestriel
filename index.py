import base64
import datetime
import imp
import io

from jupyter_dash import JupyterDash
from dash import no_update
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import dash
import pandas as pd
from app import app
from uploading import uploading_layout, overview_layout
from processing import processing_layout
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
    
    dbc.Row(dbc.Col(html.H1("NPS prediction",
                            style={"textAlign": "center"}), width=12)),
    html.Hr(),
    dcc.Store(id='raw-data'),   
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
        return []
    return html.P("This shouldn't be displayed for now...")



app.run_server()