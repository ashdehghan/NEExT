#!/usr/bin/env python3
"""
Create a single 4-panel scientific figure showing accuracy, F1, precision, and recall
across embedding dimensions for different feature sets.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
df = pd.read_csv('embedding_dimensions_20250820_220745.csv')

# Create 4-panel figure
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Accuracy', 'F1 Score', 'Precision', 'Recall'),
    vertical_spacing=0.12,
    horizontal_spacing=0.10
)

# Define colors and styles
colors = {
    'combined': '#1f77b4',
    'node_only': '#ff7f0e', 
    'structural_only': '#2ca02c',
    'random_baseline': '#d62728'
}

names = {
    'combined': 'Combined Features',
    'node_only': 'Node Features Only',
    'structural_only': 'Structural Features Only',
    'random_baseline': 'Random Baseline'
}

# Metrics to plot
metrics = [
    ('accuracy', 'accuracy_std', 1, 1),
    ('f1_score', 'f1_score_std', 1, 2),
    ('precision', 'precision_std', 2, 1),
    ('recall', 'recall_std', 2, 2)
]

# Calculate y-axis ranges (excluding random baseline)
y_ranges = {}
for metric, _, _, _ in metrics:
    values = df[df['model'] != 'random_baseline'][metric]
    max_val = values.max()
    padding = (max_val - 0.4) * 0.05  # 5% padding above max
    y_ranges[metric] = [0.4, max_val + padding]

# Plot each model
for model in ['combined', 'node_only', 'structural_only', 'random_baseline']:
    model_data = df[df['model'] == model]
    
    for metric, std_metric, row, col in metrics:
        if model == 'random_baseline':
            # Dashed line for baseline
            fig.add_trace(
                go.Scatter(
                    x=model_data['dimension'],
                    y=model_data[metric],
                    mode='lines',
                    name=names[model],
                    line=dict(color=colors[model], width=2, dash='dash'),
                    showlegend=(row == 1 and col == 1)
                ),
                row=row, col=col
            )
        else:
            # Lines with markers and error bars for actual models
            fig.add_trace(
                go.Scatter(
                    x=model_data['dimension'],
                    y=model_data[metric],
                    error_y=dict(
                        type='data',
                        array=model_data[std_metric],
                        visible=True,
                        thickness=1,
                        width=3
                    ),
                    mode='lines+markers',
                    name=names[model],
                    line=dict(color=colors[model], width=2.5),
                    marker=dict(size=7),
                    showlegend=(row == 1 and col == 1)
                ),
                row=row, col=col
            )

# Update axes
for metric, _, row, col in metrics:
    fig.update_yaxes(
        range=y_ranges[metric],
        showgrid=True,
        gridcolor='lightgray',
        row=row, col=col
    )
    fig.update_xaxes(
        range=[3, 52],
        showgrid=True,
        gridcolor='lightgray',
        row=row, col=col
    )

# Add axis labels
fig.update_xaxes(title_text="Embedding Dimension", row=2, col=1)
fig.update_xaxes(title_text="Embedding Dimension", row=2, col=2)
fig.update_yaxes(title_text="Score", row=1, col=1)
fig.update_yaxes(title_text="Score", row=2, col=1)

# Update layout for scientific appearance
fig.update_layout(
    title=dict(
        text="Reddit Binary Classification Performance Across Embedding Dimensions",
        font=dict(size=18, family="Arial")
    ),
    height=800,
    width=1100,
    font=dict(size=12, family="Arial"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Save the figure
fig.write_html("performance_metrics.html")
print("Created performance_metrics.html")