import os
import sys
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

from src.postprocessing.results_util import load_experiments_results

sys.path.append(os.getcwd())

# Sample data
np.random.seed(42)
image_types = ['Entire']
base_models = \
    ['googlenet',
     'resnet50_ImageNet_ChestX-ray14',
     'mobilenet_v2',
     'densenet161',
     'shufflenet_v2_x0_5',
     'shufflenet_v2_x1_0',
     'efficientnet_lite0',
     'densenet201',
     'densenet121',
     'efficientnet_b0',
     'densenet121_CXR',
     'efficientnet_es_pruned',
     'resnet50_ChexPert',
     'resnet34',
     'densenet169',
     'efficientnet_b1_pruned',
     'resnet101',
     'resnext50_32x4d',
     'wide_resnet50_2',
     'resnet50',
     'resnet50_ImageNet_ChexPert',
     'efficientnet_es',
     'resnet50_ChestX-ray14',
     'resnet18']

metrics = ['Accuracy', 'F1_score', 'Accuracy-Severe', 'Accuracy-Mild', 'ROC-AUC']
models = base_models
experiments = [
               'BASELINE_1release',
               'BASELINE_3release',
               'BASELINE_2release',
               'MULTI_1release_brixia_Global',
               'MULTI_1release_brixia_Lung',
               'MULTI_1release_regression',
               'MULTI_2release_brixia_Global',
               'MULTI_2release_brixia_Lung',
               'MULTI_2release_regression',
               'MULTI_3release_brixia_Global',
               'MULTI_3release_brixia_Lung',
               'MULTI_3release_regression', ]  # 'BINA_BX_cl', ,
# 'Curriculum_BINA_adaptive-2',
categories = ['mean', 'std']
cross_val_strategies = ['CV5', 'loCo']
head_modes = ['parallel', 'baseline']

df = load_experiments_results(['reports'], experiments=experiments, setups_cv=cross_val_strategies, image_types=image_types, head_modes=['parallel', 'baseline'],
                              metrics=metrics, models_names=base_models)

label_font_STYLE = {'font-weight': 'bold', 'font-size': '40px', 'margin-bottom': '40px'}
# App setup

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Multi - Model Visualization", style={'text-align': 'center', 'font-family': 'Arial, sans-serif', 'font-weight': 'bold', 'font-size': '60px'}),

    html.Div([
        html.Label('Select Metric:', style=label_font_STYLE),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[{'label': metric, 'value': metric} for metric in metrics],
            value=df['Metric'].unique()[0],  # default value
            style={'font-size': '30px'},
        ),
    ], style={'border': '3px solid #ccc', 'padding': '30px', 'margin-bottom': '40px'}),

    html.Div([
        html.Div([
            html.Label('Select Image Type:', style=label_font_STYLE),
            dcc.RadioItems(
                id='image-type-radio',
                options=[{'label': image_type, 'value': image_type} for image_type in image_types],
                value=image_types[0],  # default value
                style={'font-size': '30px'},
            ),
        ], style={'border': '10px solid #ccc', 'padding': '40px', 'margin-right': '40px', 'display': 'inline-block'}),

        html.Div([
            html.Label('Select Category:', style=label_font_STYLE),
            dcc.RadioItems(
                id='category-radio',
                options=[{'label': category, 'value': category} for category in categories],
                value=categories[0],  # default value
                style={'font-size': '30px'},
            ),
        ], style={'border': '10px solid #ccc', 'padding': '30px', 'width': '30%', 'display': 'inline-block', 'margin-left': '25px'}),

        html.Div([
            html.Label('Select Cross-Val Strategy:', style=label_font_STYLE),
            dcc.RadioItems(
                id='cross-val-radio',
                options=[{'label': strategy, 'value': strategy} for strategy in cross_val_strategies],
                value=cross_val_strategies[0],  # default value
                style={'font-size': '30px'},
            ),
        ], style={'border': '10px solid #ccc', 'padding': '40px', 'margin-right': '40px', 'display': 'inline-block'}),
    ], style={'margin-bottom': '40px'}),


    html.H2("Multiple Experiments Visualization [ 1 Release AFC ]", style={'text-align': 'center', 'font-family': 'Arial, sans-serif', 'font-weight': 'bold', 'font-size': '40px'}),
    html.Div([
        html.Label('Select Experiments:', style=label_font_STYLE),
        dcc.Checklist(
            id='experiments-checklist-1',
            options=[{'label': experiment, 'value': experiment} for experiment in [experiment for experiment in experiments if '1release' in experiment]],
            value=experiments,  # default value set to all experiments
            inline=True,  # display inline
            style={'font-size': '30px'},
        ),
    ], style={'border': '1px solid #ccc', 'padding': '20px', 'margin-bottom': '20px'}),

    html.Div([
        html.Label('Select Head Modes:', style=label_font_STYLE),
        dcc.Checklist(
            id='head-modes-checklist-1',
            options=[{'label': head_mode, 'value': head_mode} for head_mode in head_modes],
            value=head_modes,  # default value set to all head modes
            style={'font-size': '20px'},
            inline=True,  # display inline
        ),
    ], style={'border': '1px solid #ccc', 'padding': '20px', 'margin-bottom': '20px'}),

    dcc.Graph(id='scatter-plot-1', style={'height': 1000, 'width': 2500}),

    html.H2("Multiple Experiments Visualization [ 2 Release AFC ]", style={'text-align': 'center', 'font-family': 'Arial, sans-serif', 'font-weight': 'bold', 'font-size': '40px'}),
    html.Div([
        html.Label('Select Experiments:', style=label_font_STYLE),
        dcc.Checklist(
            id='experiments-checklist',
            options=[{'label': experiment, 'value': experiment} for experiment in [experiment for experiment in experiments if '2release' in experiment]],
            value=experiments,  # default value set to all experiments
            inline=True,  # display inline
            style={'font-size': '30px'},
        ),
    ], style={'border': '1px solid #ccc', 'padding': '20px', 'margin-bottom': '20px'}),

    html.Div([
        html.Label('Select Head Modes:', style=label_font_STYLE),
        dcc.Checklist(
            id='head-modes-checklist',
            options=[{'label': head_mode, 'value': head_mode} for head_mode in head_modes],
            value=head_modes,  # default value set to all head modes
            style={'font-size': '20px'},
            inline=True,  # display inline
        ),
    ], style={'border': '1px solid #ccc', 'padding': '20px', 'margin-bottom': '20px'}),

    dcc.Graph(id='scatter-plot-2', style={'height': 1000, 'width': 2500}),


    html.H2("Multiple Experiments Visualization [ 3 Release AFC ]", style={'text-align': 'center', 'font-family': 'Arial, sans-serif', 'font-weight': 'bold', 'font-size': '40px'}),
    html.Div([
        html.Label('Select Experiments:', style=label_font_STYLE),
        dcc.Checklist(
            id='experiments-checklist-3',
            options=[{'label': experiment, 'value': experiment} for experiment in [experiment for experiment in experiments if '3release' in experiment]],
            value=experiments,  # default value set to all experiments
            inline=True,  # display inline
            style={'font-size': '30px'},
        ),
    ], style={'border': '1px solid #ccc', 'padding': '20px', 'margin-bottom': '20px'}),

    html.Div([
        html.Label('Select Head Modes:', style=label_font_STYLE),
        dcc.Checklist(
            id='head-modes-checklist-3',
            options=[{'label': head_mode, 'value': head_mode} for head_mode in head_modes],
            value=head_modes,  # default value set to all head modes
            style={'font-size': '20px'},
            inline=True,  # display inline
        ),
    ], style={'border': '1px solid #ccc', 'padding': '20px', 'margin-bottom': '20px'}),

    dcc.Graph(id='scatter-plot-3', style={'height': 1000, 'width': 2500})







], style={'font-family': 'Arial, sans-serif', 'padding': '20px'})

@app.callback(
    Output('model-checklist', 'value'),
    Input('select-all-button', 'n_clicks'),
    State('model-checklist', 'value')
)
def select_deselect_all(n_clicks, selected_models):
    if n_clicks % 2 == 1:  # If odd number of clicks, deselect all
        return []
    else:  # If even number of clicks, select all
        return models


@app.callback(
    Output('scatter-plot-1', 'figure'),
    [Input('image-type-radio', 'value'),
     Input('category-radio', 'value'),
     Input('cross-val-radio', 'value'),
     Input('experiments-checklist-1', 'value'),
     Input('head-modes-checklist-1', 'value'),
     Input('metric-dropdown', 'value')]  # Add the metric input
)
def update_scatter_plot_1(selected_image_type, selected_category, selected_cross_val_strategy, selected_experiments, selected_head_modes, selected_metric):
    filtered_df = df[(df['Image Type'] == selected_image_type) &
                     (df['Category'] == selected_category) &
                     (df['Cross Val Strategy'] == selected_cross_val_strategy) &
                     (df['Experiment'].isin(selected_experiments)) &
                     (df['Head Mode'].isin(selected_head_modes)) &
                     (df['Metric'] == selected_metric) & (df['Release'] == 'AFC1')]
    # Create the scatter plot
    fig = go.Figure()

    # Define marker styles
    marker_styles = {
        'parallel': dict(symbol='circle', size=15, line=dict(width=2)),
        'serial': dict(symbol='square', size=15, line=dict(width=2))
    }

    # Find the best performance for each model
    best_performances = filtered_df.loc[filtered_df.groupby('Model')['Value'].idxmax()] if selected_category == 'mean' else filtered_df.loc[filtered_df.groupby('Model')['Value'].idxmin()]

    # Iterate through experiments and head modes to create scatter plots
    for experiment in selected_experiments:
        for head_mode in selected_head_modes:
            exp_head_df = filtered_df[(filtered_df['Experiment'] == experiment) & (filtered_df['Head Mode'] == head_mode)]
            if not exp_head_df.empty:
                # Check if the current subset includes the best performance for each model
                sizes = [22 if row['Model'] in best_performances['Model'].values and row['Value'] == best_performances[best_performances['Model'] == row['Model']]['Value'].values[0] else 12
                         for _, row in exp_head_df.iterrows()]
                # Apply the marker style based on the head mode
                marker_style = marker_styles.get(head_mode, dict(size=15, line=dict(width=2)))
                marker_style['size'] = sizes
                # Add the scatter plot for the current subset
                fig.add_trace(go.Scatter(
                    x=exp_head_df['Model'],
                    y=exp_head_df['Value'],
                    mode='markers',
                    name=f"{experiment} - {head_mode}",
                    marker=marker_style,
                ))

    # Update the layout
    fig.update_layout(
        title=f'<b>[METRIC: {selected_metric}] Scatter Plot for Selected Experiments and Head Modes ({selected_category}) - {selected_cross_val_strategy}',
        xaxis_title='Model',
        yaxis_title=f'{selected_metric}',
        legend_title='Experiment - Head Mode',
        font=dict(size=25)
    )

    return fig
@app.callback(
    Output('scatter-plot-2', 'figure'),
    [Input('image-type-radio', 'value'),
     Input('category-radio', 'value'),
     Input('cross-val-radio', 'value'),
     Input('experiments-checklist', 'value'),
     Input('head-modes-checklist', 'value'),
     Input('metric-dropdown', 'value')]  # Add the metric input
)
def update_scatter_plot_2(selected_image_type, selected_category, selected_cross_val_strategy, selected_experiments, selected_head_modes, selected_metric):
    filtered_df = df[(df['Image Type'] == selected_image_type) &
                     (df['Category'] == selected_category) &
                     (df['Cross Val Strategy'] == selected_cross_val_strategy) &
                     (df['Experiment'].isin(selected_experiments)) &
                     (df['Head Mode'].isin(selected_head_modes)) &
                     (df['Metric'] == selected_metric) & (df['Release'] == 'AFC12')]
    # Create the scatter plot
    fig = go.Figure()

    # Define marker styles
    marker_styles = {
        'parallel': dict(symbol='circle', size=15, line=dict(width=2)),
        'serial': dict(symbol='square', size=15, line=dict(width=2))
    }

    # Find the best performance for each model
    best_performances = filtered_df.loc[filtered_df.groupby('Model')['Value'].idxmax()] if selected_category == 'mean' else filtered_df.loc[filtered_df.groupby('Model')['Value'].idxmin()]

    # Iterate through experiments and head modes to create scatter plots
    for experiment in selected_experiments:
        for head_mode in selected_head_modes:
            exp_head_df = filtered_df[(filtered_df['Experiment'] == experiment) & (filtered_df['Head Mode'] == head_mode)]
            if not exp_head_df.empty:
                # Check if the current subset includes the best performance for each model
                sizes = [22 if row['Model'] in best_performances['Model'].values and row['Value'] == best_performances[best_performances['Model'] == row['Model']]['Value'].values[0] else 12
                         for _, row in exp_head_df.iterrows()]
                # Apply the marker style based on the head mode
                marker_style = marker_styles.get(head_mode, dict(size=15, line=dict(width=2)))
                marker_style['size'] = sizes
                # Add the scatter plot for the current subset
                fig.add_trace(go.Scatter(
                    x=exp_head_df['Model'],
                    y=exp_head_df['Value'],
                    mode='markers',
                    name=f"{experiment} - {head_mode}",
                    marker=marker_style,
                ))

    # Update the layout
    fig.update_layout(
        title=f'<b>[METRIC: {selected_metric}] Scatter Plot for Selected Experiments and Head Modes ({selected_category}) - {selected_cross_val_strategy}',
        xaxis_title='Model',
        yaxis_title=f'{selected_metric}',
        legend_title='Experiment - Head Mode',
        font=dict(size=25)
    )

    return fig

@app.callback(
    Output('scatter-plot-3', 'figure'),
    [Input('image-type-radio', 'value'),
     Input('category-radio', 'value'),
     Input('cross-val-radio', 'value'),
     Input('experiments-checklist-3', 'value'),
     Input('head-modes-checklist-3', 'value'),
     Input('metric-dropdown', 'value')]  # Add the metric input
)
def update_scatter_plot_3(selected_image_type, selected_category, selected_cross_val_strategy, selected_experiments, selected_head_modes, selected_metric):
    filtered_df = df[(df['Image Type'] == selected_image_type) &
                     (df['Category'] == selected_category) &
                     (df['Cross Val Strategy'] == selected_cross_val_strategy) &
                     (df['Experiment'].isin(selected_experiments)) &
                     (df['Head Mode'].isin(selected_head_modes)) &
                     (df['Metric'] == selected_metric) & (df['Release'] == 'AFC123')]
    # Create the scatter plot
    fig = go.Figure()
    # Define marker styles
    marker_styles = {
        'parallel': dict(symbol='circle', size=15, line=dict(width=2)),
        'serial': dict(symbol='square', size=15, line=dict(width=2))
    }

    # Find the best performance for each model
    best_performances = filtered_df.loc[filtered_df.groupby('Model')['Value'].idxmax()] if selected_category == 'mean' else filtered_df.loc[filtered_df.groupby('Model')['Value'].idxmin()]

    # Iterate through experiments and head modes to create scatter plots
    for experiment in selected_experiments:
        for head_mode in selected_head_modes:
            exp_head_df = filtered_df[(filtered_df['Experiment'] == experiment) & (filtered_df['Head Mode'] == head_mode)]
            if not exp_head_df.empty:
                # Check if the current subset includes the best performance for each model
                sizes = [22 if row['Model'] in best_performances['Model'].values and row['Value'] == best_performances[best_performances['Model'] == row['Model']]['Value'].values[0] else 12
                         for _, row in exp_head_df.iterrows()]
                # Apply the marker style based on the head mode
                marker_style = marker_styles.get(head_mode, dict(size=15, line=dict(width=2)))
                marker_style['size'] = sizes
                # Add the scatter plot for the current subset
                fig.add_trace(go.Scatter(
                    x=exp_head_df['Model'],
                    y=exp_head_df['Value'],
                    mode='markers',
                    name=f"{experiment} - {head_mode}",
                    marker=marker_style,
                ))

    # Update the layout
    fig.update_layout(
        title=f'<b>[METRIC: {selected_metric}] Scatter Plot for Selected Experiments and Head Modes ({selected_category}) - {selected_cross_val_strategy}',
        xaxis_title='Model',
        yaxis_title=f'{selected_metric}',
        legend_title='Experiment - Head Mode',
        font=dict(size=25)
    )

    return fig
if __name__ == '__main__':
    app.run_server(debug=False)
