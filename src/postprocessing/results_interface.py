import os
import sys
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

from src.postprocessing.results_util import load_experiments_results

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "CMC_utils"))

# Sample data
np.random.seed(42)
image_types = ['Entire', 'LungMask']
base_models = ["densenet121_CXR",
               "densenet121",
               "googlenet",
               "mobilenet_v2",
               "resnet18",
               "resnet34",
               "resnet50",
               "resnet50_ChestX-ray14",
               "resnet50_ChexPert",
               "resnet50_ImageNet_ChestX-ray14",
               "resnet50_ImageNet_ChexPert",
               "resnet101",
               "resnext50_32x4d",
               "shufflenet_v2_x0_5",
               "shufflenet_v2_x1_0",
               "wide_resnet50_2",
               ]

metrics = ['Accuracy', 'F1_score',  'Accuracy-Severe', 'Accuracy-Mild']
models = base_models
experiments = ['BINA_h1', 'BINA_BX',  'BASELINE_BINA', 'Curriculum_BINA_CL'] # 'BINA_BX_cl',
categories = ['mean', 'std']
cross_val_strategies = ['CV5', 'LoCo']
head_modes = ['parallel', 'serial', 'baseline']


df = load_experiments_results(['reports'], experiments=experiments, setups_cv=['CV5', 'LoCo'], image_types=['Entire', 'LungMask'], head_modes=['parallel', 'serial', 'baseline'] ,
                              metrics=['F1_score', 'Accuracy', 'Accuracy-Severe', 'Accuracy-Mild'], models_names=base_models)


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
            style = {'font-size': '30px'},
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
        ], style={'border': '3px solid #ccc', 'padding': '40px', 'margin-right': '40px', 'display': 'inline-block'}),

        html.Div([
            html.Label('Select Experiment:', style=label_font_STYLE),
            dcc.RadioItems(
                id='experiment-radio',
                options=[{'label': experiment, 'value': experiment} for experiment in experiments],
                value=experiments[0],  # default value
                style={'font-size': '30px'},
            ),
        ], style={'border': '3px solid #ccc', 'padding': '40px', 'margin-right': '40px', 'display': 'inline-block'}),

        html.Div([
            html.Label('Select Cross-Val Strategy:', style=label_font_STYLE),
            dcc.RadioItems(
                id='cross-val-radio',
                options=[{'label': strategy, 'value': strategy} for strategy in cross_val_strategies],
                value=cross_val_strategies[0],  # default value
                style={'font-size': '30px'},
            ),
        ], style={'border': '3px solid #ccc', 'padding': '40px', 'margin-right': '40px', 'display': 'inline-block'}),
    ], style={'margin-bottom': '40px'}),
    html.Div([
        html.Label('Select Models:', style=label_font_STYLE),
        dcc.Checklist(
            id='model-checklist',
            options=[{'label': model, 'value': model} for model in models],
            value=models,  # default value set to all models
            inline=True,  # display inline
            labelStyle={'margin-bottom': '20px', 'font-size': '30px'},
            style={'font-size': '30px'},
        ),
        html.Button('Select/Deselect All', id='select-all-button', n_clicks=0),
    ], style={'border': '1px solid #ccc', 'padding': '40px', 'width': '100%', 'font-size': '25px'}),
    html.Div([



        html.Div([
            html.Label('Select Category:', style=label_font_STYLE),
            dcc.RadioItems(
                id='category-radio',
                options=[{'label': category, 'value': category} for category in categories],
                value=categories[0],  # default value
                style={'font-size': '30px'},
            ),
        ], style={'border': '10px solid #ccc', 'padding': '20px', 'width': '30%', 'display': 'inline-block', 'margin-left': '25px'}),
        html.Div([
            html.Label('Select Head Mode:', style=label_font_STYLE),
            dcc.RadioItems(
                id='head-mode-radio',
                options=[{'label': head_mode, 'value': head_mode} for head_mode in head_modes],
                value=head_modes[0],  # default value
                style={'font-size': '30px'},
            ),
        ], style={'border': '10px solid #ccc', 'padding': '20px', 'width': '30%', 'display': 'inline-block', 'margin-left': '25px'}),
    ], style={'margin-bottom': '100px'}),



    dcc.Graph(id='plot', style={'height': 1000, 'width': 2500}),
    html.H2("Multiple Experiments Visualization", style={'text-align': 'center', 'font-family': 'Arial, sans-serif', 'font-weight': 'bold'}),
    html.Div([
        html.Label('Select Experiments:', style=label_font_STYLE),
        dcc.Checklist(
            id='experiments-checklist',
            options=[{'label': experiment, 'value': experiment} for experiment in experiments],
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

    dcc.Graph(id='scatter-plot', style={'height': 1000, 'width': 2500})
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
    Output('plot', 'figure'),
    [Input('image-type-radio', 'value'),
     Input('model-checklist', 'value'),
     Input('experiment-radio', 'value'),
     Input('category-radio', 'value'),
     Input('cross-val-radio', 'value'),
     Input('head-mode-radio', 'value'),
     Input('metric-dropdown', 'value')]  # Add the metric input
)
def update_bar_plot(selected_image_type, selected_models, selected_experiment, selected_category, selected_cross_val_strategy, selected_head_mode, selected_metric):
    filtered_df = df[(df['Image Type'] == selected_image_type) &
                     (df['Model'].isin(selected_models)) &
                     (df['Experiment'] == selected_experiment) &
                     (df['Category'] == selected_category) &
                     (df['Cross Val Strategy'] == selected_cross_val_strategy) &
                     (df['Head Mode'] == selected_head_mode) &
                     (df['Metric'] == selected_metric)]
    fig = px.bar(filtered_df, x='Model', y='Value', color='Model', hover_data=['Value'],
                 title=f'Bar Plot for {selected_experiment} ({selected_category}) - {selected_cross_val_strategy} - {selected_head_mode}',
    )

    # Update text font size inside bars
    fig.update_traces(textfont_size=25)
    # Update font size for axis titles, tick labels and legend
    fig.update_layout(
        title_text=f'Show Single Experiment: {selected_experiment}',
        title_font=dict(size=35),
        xaxis=dict(
            title_text='Model Name',
            title_font=dict(size=27),
            tickfont=dict(size=23),
            showgrid=True,  # Show gridlines for x-axis
            gridwidth=1,  # Set gridline width for x-axis
            gridcolor='lightgrey',  # Set gridline color for x-axis
        ),
        yaxis=dict(
            title_text=f'{selected_metric}',
            title_font=dict(size=27),
            tickfont=dict(size=20),
            showgrid=True,  # Show gridlines for y-axis
            gridwidth=2,  # Set gridline width for y-axis
            gridcolor='black',  # Set gridline color for y-axis
            dtick = 2 if 'Accuracy' in selected_metric else 1
        ),
        plot_bgcolor='white',  # Set the background color of the plot
        legend=dict(
            font=dict(size=24)
        )
    )

    return fig


@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('image-type-radio', 'value'),
     Input('category-radio', 'value'),
     Input('cross-val-radio', 'value'),
     Input('experiments-checklist', 'value'),
     Input('head-modes-checklist', 'value'),
     Input('metric-dropdown', 'value')]  # Add the metric input
)
def update_scatter_plot(selected_image_type, selected_category, selected_cross_val_strategy, selected_experiments, selected_head_modes, selected_metric):
    filtered_df = df[(df['Image Type'] == selected_image_type) &
                     (df['Category'] == selected_category) &
                     (df['Cross Val Strategy'] == selected_cross_val_strategy) &
                     (df['Experiment'].isin(selected_experiments)) &
                     (df['Head Mode'].isin(selected_head_modes)) &
                     (df['Metric'] == selected_metric)]
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
    app.run_server(debug=True)
