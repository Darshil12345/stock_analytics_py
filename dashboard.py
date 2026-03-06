"""
dashboard.py — Dash application factory.

Creates the Dash instance, defines all page-renderer functions,
and registers every callback module.  Import `create_dash_app()`
from main.py to obtain the app object.
"""
from __future__ import annotations
from datetime import datetime, timedelta

from dash import Dash, dcc, html, Input, Output, State, ALL, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash

import state
from config import (
    DEFAULT_INDICES, ALL_AVAILABLE_INDICES, EXTERNAL_LINKS,
    PROB_CUTOFF_DEFAULT,
)
from models.model_registry import MODEL_MAP

# ── CSS / index template ──────────────────────────────────────────────────────
_INDEX_STRING = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}<title>Stock Analytics Pro</title>{%favicon%}{%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
            body { font-family:'Poppins',sans-serif!important; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); min-height:100vh; padding:20px; }
            h2,h3,h4,h5{font-weight:600;color:#2d3748;}
            .page-title{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.2rem;font-weight:700;text-align:center;margin-bottom:30px;}
            .card{border-radius:15px!important;box-shadow:0 10px 30px rgba(0,0,0,.15)!important;border:none!important;margin-bottom:25px;}
            .card-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%)!important;color:white!important;border:none!important;padding:20px!important;font-weight:600;}
            .card-body{padding:25px!important;}
            .btn{border-radius:25px!important;padding:10px 25px!important;font-weight:500;transition:all .3s;border:none!important;box-shadow:0 4px 15px rgba(0,0,0,.2);}
            .btn:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,.3);}
            .btn-primary{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%)!important;}
            .btn-success{background:linear-gradient(135deg,#56ab2f 0%,#a8e063 100%)!important;}
            .btn-info{background:linear-gradient(135deg,#2193b0 0%,#6dd5ed 100%)!important;}
            .btn-warning{background:linear-gradient(135deg,#f09819 0%,#edde5d 100%)!important;}
            .section-header{font-size:1.2rem;font-weight:600;color:#667eea;margin:20px 0 12px;border-left:4px solid #667eea;padding-left:12px;}
            .sidebar-link{color:rgba(255,255,255,.8)!important;padding:15px 20px!important;margin:5px 0!important;border-radius:10px!important;transition:all .3s!important;font-weight:500!important;border:none!important;}
            .sidebar-link:hover{background:rgba(255,255,255,.15)!important;color:white!important;transform:translateX(5px);}
            .sidebar-link.active{background:white!important;color:#667eea!important;box-shadow:0 4px 15px rgba(0,0,0,.2);}
            .external-link-card{transition:all .3s;cursor:pointer;border:2px solid #e2e8f0;}
            .external-link-card:hover{transform:translateY(-5px);box-shadow:0 10px 25px rgba(102,126,234,.3)!important;border-color:#667eea;}
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>'''


# ── App factory ───────────────────────────────────────────────────────────────
def create_dash_app() -> Dash:
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True,
    )
    app.index_string = _INDEX_STRING
    app.layout = _build_layout()
    _register_all_callbacks(app)
    return app


# ── Layout ────────────────────────────────────────────────────────────────────
def _build_layout():
    sidebar = html.Div([
        html.Div([
            html.I(className="fas fa-chart-line fa-3x mb-3", style={"color":"white"}),
            html.H4("Stock Analytics", style={"color":"white","fontWeight":"700"}),
            html.P("Enhanced Platform", style={"color":"rgba(255,255,255,.8)","fontSize":".9rem"}),
        ], style={"textAlign":"center","padding":"30px 20px","borderBottom":"2px solid rgba(255,255,255,.2)"}),
        html.Div(dbc.Nav([
            _nav_link("nav-config",    "fa-cog",              "Configuration"),
            _nav_link("nav-master",    "fa-database",         "Master Data"),
            _nav_link("nav-eda",       "fa-chart-bar",        "EDA"),
            _nav_link("nav-ml",        "fa-brain",            "ML Prediction"),
            _nav_link("nav-sentiment", "fa-comments",         "Sentiment"),
            _nav_link("nav-external",  "fa-external-link-alt","External Links"),
        ], vertical=True, pills=True, className="flex-column"),
        style={"padding":"20px 10px"}),
    ], style={
        "position":"fixed","top":0,"left":0,"bottom":0,"width":"280px",
        "background":"linear-gradient(180deg,#667eea 0%,#764ba2 100%)",
        "overflowY":"auto","boxShadow":"4px 0 20px rgba(0,0,0,.1)","zIndex":1000,
    })

    return dbc.Container([
        dcc.Store(id="app-mode-store",       data="default"),
        dcc.Store(id="selected-indices-store", data=list(DEFAULT_INDICES.keys())),
        dcc.Store(id="selected-features-store", data=["OHLC","Returns","Ratios","Time"]),
        dcc.Store(id="derived-features-store",  data=[]),
        dcc.Store(id="current-page-store",      data="config"),
        dbc.Row([
            dbc.Col(sidebar, width=0, style={"padding":0}),
            dbc.Col(html.Div([
                html.H2("📊 Enhanced Global Stock Market Analytics", className="page-title"),
                html.Hr(),
                html.Div(id="page-content"),
                dcc.Download(id="download-master"),
            ], style={"marginLeft":"300px","padding":"20px"}), width=12),
        ], className="g-0"),
    ], fluid=True, style={"padding":0,"margin":0})


def _nav_link(link_id, icon, label):
    return dbc.NavLink([
        html.I(className=f"fas {icon} me-3"), html.Span(label),
    ], id=link_id, href="#", className="sidebar-link")


# ── Page renderers ────────────────────────────────────────────────────────────
def render_config_page(mode, sel_indices, sel_features, der_features):
    return dbc.Card([
        dbc.CardHeader([html.I(className="fas fa-cog me-2"), "Application Configuration"]),
        dbc.CardBody([
            html.H5("🎯 Select Mode", className="section-header"),
            dbc.RadioItems(id="mode-selector", options=[
                {"label": " Default Mode (7 hardcoded indices)", "value": "default"},
                {"label": " Customize Mode (choose indices & features)", "value": "customize"},
            ], value=mode),
            html.Hr(),
            html.Div(id="customize-options", style={"display":"block" if mode=="customize" else "none"}, children=[
                html.H5("📈 Select Indices", className="section-header"),
                dbc.Checklist(id="indices-checklist",
                    options=[{"label": f" {k}", "value": k} for k in ALL_AVAILABLE_INDICES],
                    value=sel_indices, style={"columnCount":3}),
                html.Hr(),
                html.H5("🔧 Select Features", className="section-header"),
                dbc.Checklist(id="features-checklist", options=[
                    {"label": " OHLC",                 "value": "OHLC"},
                    {"label": " Returns (daily %)",     "value": "Returns"},
                    {"label": " Ratios (Open/Close)",   "value": "Ratios"},
                    {"label": " Time Features",         "value": "Time"},
                    {"label": " Volume",                "value": "Volume"},
                ], value=sel_features),
                html.Hr(),
                html.H5("⚡ Derived Features", className="section-header"),
                html.P("Examples: NIFTY_MA20 = MA(NIFTY_Close, 20)  |  NIFTY_Range = NIFTY_High - NIFTY_Low",
                       className="text-muted", style={"fontSize":".9rem"}),
                dbc.Row([
                    dbc.Col(dcc.Input(id="new-formula-input", type="text",
                                      placeholder="NIFTY_MA50 = MA(NIFTY_Close, 50)",
                                      style={"width":"100%"}), width=9),
                    dbc.Col(dbc.Button([html.I(className="fas fa-plus me-2"),"Add"],
                                        id="add-formula-btn", color="success", className="w-100"), width=3),
                ]),
                html.Div(id="formula-list", className="mt-3"),
                html.Hr(),
                html.H5("➕ Add Custom Index", className="section-header"),
                dbc.Row([
                    dbc.Col(dcc.Input(id="new-index-name", type="text",
                                      placeholder="Short name e.g. MYIDX", style={"width":"100%"}), width=5),
                    dbc.Col(dcc.Input(id="new-index-ticker", type="text",
                                      placeholder="Ticker e.g. ^MYIDX", style={"width":"100%"}), width=5),
                    dbc.Col(dbc.Button([html.I(className="fas fa-plus me-2"),"Add"],
                                        id="add-index-btn", color="secondary", className="w-100"), width=2),
                ]),
                html.Div(id="add-index-status", className="mt-2"),
            ]),
            html.Hr(),
            dbc.Button([html.I(className="fas fa-save me-2"),"Save Configuration"],
                        id="save-config-btn", color="primary", size="lg", className="w-100"),
        ]),
    ])


def render_page1(mode, sel_indices, sel_features):
    return dbc.Card([
        dbc.CardHeader([html.I(className="fas fa-database me-2"),
                         f"Master Data Builder ({mode.title()} Mode)"]),
        dbc.CardBody([
            dbc.Alert([html.I(className="fas fa-info-circle me-2"),
                        f"Indices: {', '.join(sel_indices)} | Features: {', '.join(sel_features)}"],
                       color="info"),
            dbc.Row([
                dbc.Col([html.Label("📅 Start Date"),
                          dcc.DatePickerSingle(id="start-date", date="2018-01-01", display_format="DD/MM/YYYY")], width=3),
                dbc.Col([html.Label("📅 End Date"),
                          dcc.DatePickerSingle(id="end-date", date="2024-12-31", display_format="DD/MM/YYYY")], width=3),
                dbc.Col([html.Br(), dbc.Button([html.I(className="fas fa-play me-2"),"Build Master"],
                          id="btn-build", color="primary", className="w-100")], width=2),
                dbc.Col([html.Br(), dbc.Button([html.I(className="fas fa-save me-2"),"Save CSV"],
                          id="btn-save", color="success", className="w-100")], width=2),
                dbc.Col([html.Br(), dbc.Button([html.I(className="fas fa-download me-2"),"Download"],
                          id="btn-download", color="info", className="w-100")], width=2),
            ]),
            html.Hr(),
            html.Div(id="p1-status"), html.Div(id="p1-save-status"), html.Div(id="p1-preview"),
        ]),
    ])


def render_page2():
    si = state.SELECTED_INDICES
    opts = [{"label": k, "value": k} for k in si]
    return dbc.Card([
        dbc.CardHeader([html.I(className="fas fa-chart-line me-2"), "Exploratory Data Analysis"]),
        dbc.CardBody([
            # Univariate
            html.H4("📊 Univariate Analysis", className="section-header"),
            dbc.Row([
                dbc.Col([html.Label("Index"),
                          dcc.Dropdown(id="uni-index", options=opts, value=si[0] if si else None)], width=4),
                dbc.Col([html.Label("Plot Type"),
                          dcc.Dropdown(id="uni-plot", options=[
                              {"label":"Time Series","value":"timeseries"},
                              {"label":"Histogram + KDE","value":"hist_kde"},
                              {"label":"Box Plot by Year","value":"box"},
                              {"label":"Violin Plot","value":"violin"},
                              {"label":"Rolling Volatility","value":"rolling_vol"},
                              {"label":"Cumulative Returns","value":"cumulative"},
                              {"label":"Drawdown","value":"drawdown"},
                              {"label":"Statistical Summary","value":"stats"},
                          ], value="timeseries")], width=4),
                dbc.Col([html.Br(), dbc.Button([html.I(className="fas fa-chart-bar me-2"),"Generate"],
                          id="btn-uni", color="primary", className="w-100")], width=4),
            ]),
            html.Div(id="uni-status", className="mt-2"),
            dcc.Loading([dcc.Graph(id="uni-figure"), html.Div(id="uni-table")], type="circle"),
            html.Hr(style={"margin":"40px 0"}),

            # Bivariate
            html.H4("📈 Bivariate Analysis", className="section-header"),
            dbc.Row([
                dbc.Col([html.Label("Index X"),
                          dcc.Dropdown(id="bi-index-1", options=opts, value=si[0] if si else None)], width=3),
                dbc.Col([html.Label("Index Y"),
                          dcc.Dropdown(id="bi-index-2", options=opts, value=si[1] if len(si)>1 else None)], width=3),
                dbc.Col([html.Label("Plot Type"),
                          dcc.Dropdown(id="bi-plot", options=[
                              {"label":"Scatter","value":"scatter"},
                              {"label":"Scatter + Trendline","value":"scatter_trend"},
                              {"label":"Correlation","value":"correlation"},
                              {"label":"Lagged Correlation","value":"lagged_corr"},
                          ], value="scatter")], width=3),
                dbc.Col([html.Br(), dbc.Button([html.I(className="fas fa-chart-line me-2"),"Generate"],
                          id="btn-bi", color="primary", className="w-100")], width=3),
            ]),
            html.Div(id="bi-status", className="mt-2"),
            dcc.Loading([dcc.Graph(id="bi-figure")], type="circle"),
            html.Hr(style={"margin":"40px 0"}),

            # Multivariate
            html.H4("🔀 Multivariate Analysis", className="section-header"),
            dbc.Row([
                dbc.Col([html.Label("Indices"),
                          dcc.Dropdown(id="multi-indices", options=opts, multi=True, value=si[:4])], width=6),
                dbc.Col([html.Label("Plot Type"),
                          dcc.Dropdown(id="multi-plot", options=[
                              {"label":"Correlation Matrix","value":"corr_matrix"},
                              {"label":"Scatter Matrix","value":"scatter_matrix"},
                              {"label":"PCA","value":"pca"},
                              {"label":"Multi-index Close Compare","value":"multi_compare"},
                          ], value="corr_matrix")], width=3),
                dbc.Col([html.Br(), dbc.Button([html.I(className="fas fa-project-diagram me-2"),"Generate"],
                          id="btn-multi", color="primary", className="w-100")], width=3),
            ]),
            html.Div(id="multi-status", className="mt-2"),
            dcc.Loading([dcc.Graph(id="multi-figure")], type="circle"),
        ]),
    ])


def render_page3():
    if state.MASTER_DF is None:
        return dbc.Alert("Build master data first", color="warning")
    df = state.MASTER_DF
    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()

    return dbc.Card([
        dbc.CardHeader([html.I(className="fas fa-brain me-2"), "ML Prediction"]),
        dbc.CardBody([
            # Date range
            dbc.Row([
                dbc.Col([html.Label("📅 Train Start"), dcc.DatePickerSingle(id="train-start", date=min_d,        display_format="DD/MM/YYYY")], width=3),
                dbc.Col([html.Label("📅 Train End"),   dcc.DatePickerSingle(id="train-end",  date="2022-12-31",  display_format="DD/MM/YYYY")], width=3),
                dbc.Col([html.Label("📅 Test Start"),  dcc.DatePickerSingle(id="test-start", date="2023-01-01",  display_format="DD/MM/YYYY")], width=3),
                dbc.Col([html.Label("📅 Test End"),    dcc.DatePickerSingle(id="test-end",   date=max_d,         display_format="DD/MM/YYYY")], width=3),
            ]),
            html.Hr(),

            # Model + cutoff + explainability toggles
            dbc.Row([
                dbc.Col([html.Label("🤖 Model"),
                          dcc.Dropdown(id="ml-model",
                                        options=[{"label":k,"value":k} for k in MODEL_MAP],
                                        value="Random Forest")], width=3),
                dbc.Col([html.Label("🎯 Probability Cutoff"),
                          dcc.Slider(id="prob-cutoff", min=0.3, max=0.8, step=0.01,
                                     value=PROB_CUTOFF_DEFAULT,
                                     marks={i/100: f"{i/100:.2f}" for i in range(30,81,10)},
                                     tooltip={"placement":"bottom","always_visible":True})], width=4),
                dbc.Col([
                    html.Label("⚙️ Explainability"),
                    dbc.Checklist(id="chk-run-shap", options=[{"label":" Run SHAP","value":True}], value=[]),
                    dbc.Checklist(id="chk-run-lime", options=[{"label":" Run LIME","value":True}], value=[]),
                ], width=2),
                dbc.Col([html.Br(), dbc.Button([html.I(className="fas fa-rocket me-2"),"Train Model"],
                          id="btn-train", color="primary", size="lg", className="w-100")], width=3),
            ]),
            html.Hr(),

            html.Div(id="ml-status"),
            dcc.Loading([
                html.Div(id="ml-metrics"),
                dbc.Row([dbc.Col(dcc.Graph(id="roc-curve"), width=6),
                          dbc.Col(dcc.Graph(id="confusion-matrix"), width=6)]),
                html.Div(id="model-specific-output"),

                # ── Predictions table ─────────────────────────────────────
                html.Div(id="predictions-table"),

                html.Div(id="cr-table-container"),
                html.Div(id="shap-section"),
                html.Div(id="lime-section"),
            ], type="circle"),

            # ── Forecast section ──────────────────────────────────────────
            html.Hr(),
            dbc.Card([
                dbc.CardHeader([html.I(className="fas fa-binoculars me-2"),
                                 "🔭 Forecast — Predict Beyond Master Data"]),
                dbc.CardBody([
                    dbc.Alert(
                        "Train a model above first, then select future dates to forecast.",
                        color="info", className="mb-3",
                    ),
                    dbc.Row([
                        dbc.Col([html.Label("📅 Forecast Start"),
                                  dcc.DatePickerSingle(id="forecast-start",
                                                        date=str(max_d),
                                                        display_format="DD/MM/YYYY")], width=4),
                        dbc.Col([html.Label("📅 Forecast End"),
                                  dcc.DatePickerSingle(id="forecast-end",
                                                        date=str(max_d),
                                                        display_format="DD/MM/YYYY")], width=4),
                        dbc.Col([html.Br(),
                                  dbc.Button([html.I(className="fas fa-binoculars me-2"), "Run Forecast"],
                                              id="btn-forecast", color="success", className="w-100")], width=4),
                    ]),
                    html.Br(),
                    dcc.Loading([
                        html.Div(id="forecast-status"),
                        dcc.Graph(id="forecast-chart"),
                        html.Div(id="forecast-table"),
                    ], type="circle"),
                ]),
            ], className="mt-3"),

            # ── On-demand explainability ──────────────────────────────────
            html.Hr(),
            dbc.Card([
                dbc.CardHeader("🔬 On-Demand Explainability (after training)"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dbc.Button([html.I(className="fas fa-eye me-2"), "Run SHAP"],
                                            id="btn-shap", color="info"), width=3),
                        dbc.Col([html.Label("LIME — test sample index"),
                                  dcc.Input(id="lime-sample-idx", type="number", value=0,
                                            min=0, style={"width":"100%"})], width=3),
                        dbc.Col(dbc.Button([html.I(className="fas fa-lemon me-2"), "Run LIME"],
                                            id="btn-lime", color="warning"), width=3),
                    ]),
                    html.Br(),
                    html.Div(id="shap-ondemand-output"),
                    html.Div(id="lime-ondemand-output"),
                ]),
            ], className="mt-3"),

            # ── Model comparison ──────────────────────────────────────────
            html.Hr(),
            dbc.Card([
                dbc.CardHeader("📚 Model Comparison"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id="compare-models", multi=True,
                                              options=[{"label":k,"value":k} for k in MODEL_MAP],
                                              value=list(MODEL_MAP.keys())), width=9),
                        dbc.Col(dbc.Button([html.I(className="fas fa-chart-bar me-2"),"Compare"],
                                            id="btn-compare-models", color="warning"), width=3),
                    ]),
                    html.Br(),
                    dcc.Loading([
                        html.Div(id="model-compare-status"),
                        dcc.Graph(id="model-compare-figure"),
                        html.Div(id="model-compare-table"),
                    ], type="circle"),
                ]),
            ], className="mt-3"),
        ]),
    ])


def render_page4():
    return dbc.Card([
        dbc.CardHeader([html.I(className="fas fa-comments me-2"), "News Sentiment Analysis"]),
        dbc.CardBody([
            html.Label("🔎 Source"),
            dbc.RadioItems(id="news-source", options=[
                {"label":" Upload CSV","value":"upload"},
                {"label":" Google News","value":"google"},
            ], value="upload", inline=True),
            html.Hr(),
            html.Div(id="news-upload-area", children=[
                html.Label("📁 Upload CSV"),
                dcc.Upload(id="upload-news",
                    children=html.Div([html.I(className="fas fa-cloud-upload-alt fa-3x mb-2"),
                                        html.Br(), "Drag & Drop or ", html.A("Select CSV")]),
                    style={"border":"3px dashed #cbd5e0","borderRadius":"15px","textAlign":"center","padding":"30px"}),
            ]),
            html.Div(id="news-google-area", style={"display":"none"}, children=[
                dbc.Row([
                    dbc.Col([html.Label("📅 Start"),
                              dcc.DatePickerSingle(id="news-start",
                                                    date=(datetime.now()-timedelta(days=30)).date())], width=4),
                    dbc.Col([html.Label("📅 End"),
                              dcc.DatePickerSingle(id="news-end", date=datetime.now().date())], width=4),
                    dbc.Col([html.Label("🔎 Keyword"),
                              dcc.Input(id="news-index-term", placeholder="NIFTY, gold…",
                                        type="text", style={"width":"100%"})], width=4),
                ]),
                html.Br(),
                html.Label("🔢 Max Articles"),
                dcc.Input(id="news-max", type="number", value=200, min=10, max=1000),
            ]),
            html.Hr(),
            html.Label("🚫 Custom Stopwords (comma-separated)"),
            dcc.Input(id="custom-stopwords", type="text", placeholder="nse, sensex…",
                      style={"width":"100%"}),
            html.Hr(),
            dbc.Button([html.I(className="fas fa-play me-2"),"Analyze Sentiment"],
                        id="btn-analyze", color="primary", size="lg", className="mb-3"),
            html.Div(id="p4-status"),
            dcc.Loading([
                html.H5("☁️ Wordcloud", className="section-header"),
                html.Div(html.Img(id="wc-img", style={"maxWidth":"100%","borderRadius":"15px"}),
                          style={"textAlign":"center","marginBottom":"30px"}),
                html.H5("📊 Sentiment Distribution", className="section-header"),
                dcc.Graph(id="sentiment-dist"),
                html.H5("📈 Average Score", className="section-header"),
                html.Div(id="avg-score"),
                html.H5("🧾 Preview (first 20)", className="section-header"),
                html.Div(id="p4-preview"),
                html.H5("📋 Full Table", className="section-header"),
                html.Div(id="p4-table"),
            ], type="circle"),
        ]),
    ])


def render_external_page():
    return dbc.Card([
        dbc.CardHeader([html.I(className="fas fa-external-link-alt me-2"), "External Resources"]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-chart-line fa-3x mb-3", style={"color":"#667eea"}),
                        html.H5(name, style={"fontWeight":"600","marginBottom":"10px"}),
                        dbc.Button([html.I(className="fas fa-external-link-alt me-2"),"Visit"],
                                    href=url, target="_blank", color="primary", size="sm", className="w-100"),
                    ], style={"textAlign":"center"}),
                ])], className="external-link-card h-100"),
                width=12, md=6, lg=4, className="mb-4")
                for name, url in EXTERNAL_LINKS.items()
            ]),
        ]),
    ])


# ── Callback registration ────────────────────────────────────────────────────
def _register_all_callbacks(app):
    from callbacks import eda_callbacks, ml_callbacks, sentiment_callbacks
    eda_callbacks.register(app)
    ml_callbacks.register(app)
    sentiment_callbacks.register(app)
    _register_nav_callbacks(app)
    _register_config_callbacks(app)
    _register_data_callbacks(app)


def _register_nav_callbacks(app):
    pages = ["config", "master", "eda", "ml", "sentiment", "external"]
    nav_ids = [f"nav-{p}" for p in pages]

    @app.callback(
        [Output(nid, "active") for nid in nav_ids] + [Output("current-page-store","data")],
        [Input(nid, "n_clicks") for nid in nav_ids],
        prevent_initial_call=True,
    )
    def update_nav(*_):
        ctx = dash.callback_context
        if not ctx.triggered:
            return [True] + [False]*5 + ["config"]
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        idx = nav_ids.index(btn) if btn in nav_ids else 0
        active = [i == idx for i in range(len(nav_ids))]
        return active + [pages[idx]]

    @app.callback(
        Output("page-content", "children"),
        Input("current-page-store", "data"),
        [State("app-mode-store",           "data"),
         State("selected-indices-store",   "data"),
         State("selected-features-store",  "data"),
         State("derived-features-store",   "data")],
    )
    def render_page(page, mode, sel_idx, sel_feat, der_feat):
        if page == "config":    return render_config_page(mode, sel_idx, sel_feat, der_feat)
        if page == "master":    return render_page1(mode, sel_idx, sel_feat)
        if page == "eda":       return render_page2()
        if page == "ml":        return render_page3()
        if page == "sentiment": return render_page4()
        if page == "external":  return render_external_page()
        return html.Div("Select a page from the sidebar")


def _register_config_callbacks(app):
    @app.callback(Output("customize-options","style"), Input("mode-selector","value"))
    def toggle_customize(mode):
        return {"display":"block"} if mode == "customize" else {"display":"none"}

    @app.callback(Output("formula-list","children"), Input("derived-features-store","data"))
    def update_formulas(formulas):
        if not formulas:
            return html.P("No derived features yet", className="text-muted")
        return dbc.ListGroup([
            dbc.ListGroupItem([
                html.Span(f, style={"flex":1}),
                dbc.Button("❌", id={"type":"remove-formula","index":i},
                            color="danger", size="sm", className="ms-2"),
            ], className="d-flex justify-content-between align-items-center")
            for i, f in enumerate(formulas)
        ])

    @app.callback(
        Output("derived-features-store","data"),
        [Input("add-formula-btn","n_clicks"),
         Input({"type":"remove-formula","index":ALL},"n_clicks")],
        [State("new-formula-input","value"), State("derived-features-store","data")],
        prevent_initial_call=True,
    )
    def manage_formulas(add_clicks, remove_clicks, new_f, current):
        ctx = dash.callback_context
        if not ctx.triggered: raise PreventUpdate
        trigger = ctx.triggered[0]["prop_id"]
        if "add-formula-btn" in trigger and new_f:
            if new_f not in current: current.append(new_f)
        elif "remove-formula" in trigger:
            idx = eval(trigger.split(".")[0])["index"]
            if 0 <= idx < len(current): current.pop(idx)
        return current

    @app.callback(
        [Output("app-mode-store","data"),
         Output("selected-indices-store","data"),
         Output("selected-features-store","data")],
        Input("save-config-btn","n_clicks"),
        [State("mode-selector","value"),
         State("indices-checklist","value"),
         State("features-checklist","value")],
        prevent_initial_call=True,
    )
    def save_config(_, mode, indices, features):
        state.APP_MODE = mode
        if mode == "default":
            state.SELECTED_INDICES  = list(DEFAULT_INDICES.keys())
            state.SELECTED_FEATURES = ["OHLC","Returns","Ratios","Time"]
        else:
            state.SELECTED_INDICES  = indices  or list(DEFAULT_INDICES.keys())
            state.SELECTED_FEATURES = features or ["OHLC","Returns"]
        return state.APP_MODE, state.SELECTED_INDICES, state.SELECTED_FEATURES

    @app.callback(
        [Output("add-index-status","children"),
         Output("selected-indices-store","data",  allow_duplicate=True),
         Output("indices-checklist","options",     allow_duplicate=True)],
        Input("add-index-btn","n_clicks"),
        [State("new-index-name","value"),
         State("new-index-ticker","value"),
         State("selected-indices-store","data")],
        prevent_initial_call=True,
    )
    def add_custom_index(_, name, ticker, current_sel):
        if not name or not ticker:
            return dbc.Alert("Provide both name and ticker", color="warning"), dash.no_update, dash.no_update
        name, ticker = name.strip().upper(), ticker.strip()
        if name not in state.INDICES_REGISTRY:
            state.INDICES_REGISTRY[name] = ticker
        current_sel = current_sel or []
        if name not in current_sel:
            current_sel.append(name)
        opts = [{"label": f" {k}", "value": k} for k in state.INDICES_REGISTRY]
        return dbc.Alert(f"✅ Added {name} → {ticker}", color="success"), current_sel, opts


def _register_data_callbacks(app):
    from pipelines.data_pipeline import run as run_data

    @app.callback(
        [Output("p1-status","children"), Output("p1-preview","children")],
        Input("btn-build","n_clicks"),
        [State("start-date","date"), State("end-date","date"),
         State("app-mode-store","data"), State("selected-indices-store","data"),
         State("selected-features-store","data"), State("derived-features-store","data")],
        prevent_initial_call=True,
    )
    def build_master_cb(_, start, end, mode, sel_idx, sel_feat, der_feat):
        try:
            df = run_data(start, end, mode, sel_idx, sel_feat, der_feat)
            status  = dbc.Alert(f"✅ {len(df)} rows | {len(df.columns)} cols | Mode: {mode}", color="success")
            preview = dash_table.DataTable(
                columns=[{"name": c,"id": c} for c in df.columns],
                data=df.head(20).select_dtypes(include="number").round(6).combine_first(df.head(20)).to_dict("records"),
                style_table={"overflowX":"auto"},
                style_cell={"textAlign":"left","padding":"10px"},
                style_header={"backgroundColor":"#667eea","color":"white","fontWeight":"600"},
                page_size=20,
            )
            return status, preview
        except Exception as exc:
            return dbc.Alert(f"❌ {exc}", color="danger"), html.Div()

    @app.callback(Output("p1-save-status","children"), Input("btn-save","n_clicks"), prevent_initial_call=True)
    def save_master(_):
        if state.MASTER_DF is None:
            return dbc.Alert("Build master first", color="warning")
        from config import MASTER_PATH
        try:
            state.MASTER_DF.to_csv(MASTER_PATH, index=False)
            return dbc.Alert(f"✅ Saved to {MASTER_PATH}", color="success")
        except Exception as exc:
            return dbc.Alert(f"❌ {exc}", color="danger")

    @app.callback(Output("download-master","data"), Input("btn-download","n_clicks"), prevent_initial_call=True)
    def download_master(_):
        if state.MASTER_DF is None: raise PreventUpdate
        return dcc.send_data_frame(state.MASTER_DF.to_csv, "master.csv", index=False)