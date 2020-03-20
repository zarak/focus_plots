import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from src.build_features import resampled_teams
sns.set(rc={'figure.figsize':(11, 4)})


def plot_ts(data, title):
    ax = sns.lineplot(
        x=data.index,
        y=data.SWRPS,
        data=data,
        dashes=False,
        marker='o',
        label=title
    )
    ax.set(ylim=(0,1.3))
    plt.title(question)


def plot_divergence(processed, scoring_method, resampling_period):
    kiwi, kiwi_KL, mango, mango_KL = resampled_teams(processed)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                    x=kiwi.index,
                    y=kiwi[scoring_method],
                    name="Kiwis",
                    mode="markers",
                    text=[f"KL divergence in {resampling_period}: {KL}" for KL in kiwi_KL],
                    line_color='deepskyblue',
                    marker=dict(
                        size=np.nan_to_num(kiwi_KL),
                        sizemode='area',
                        sizeref=2.*max(kiwi_KL)/(40.**2),
                        sizemin=4
                    ),
                    opacity=0.8))

    fig.add_trace(go.Scatter(
                    x=mango.index,
                    y=mango[scoring_method],
                    name="Mangoes",
                    mode="markers",
                    # text=[f"Number of forecasts in {resampling_period}: {count}" for count in mango_marker_size],
                    text=[f"KL divergence in {resampling_period}: {KL}" for KL in mango_KL],
                    line_color='dimgray',
                    marker=dict(
                        # size=mango_marker_size,
                        size=np.nan_to_num(mango_KL),
                        sizemode='area',
                        sizeref=2.*max(mango_KL)/(40.**2),
                        sizemin=4
                    ),
                    opacity=0.8))

    # Mark 'interventions' or special events
    # fig.update_layout(
        # shapes=[
            # # 1st highlight during Feb 4 - Feb 6
            # go.layout.Shape(
                # type="rect",
                # # x-reference is assigned to the x-values
                # xref="x",
                # # y-reference is assigned to the plot paper [0,1]
                # yref="paper",
                # x0="2019-11-15",
                # y0=0,
                # x1="2019-11-16",
                # y1=1,
                # fillcolor="LightSalmon",
                # opacity=0.5,
                # layer="below",
                # line_width=0,
            # ),
            # # 2nd highlight during Feb 20 - Feb 23
            # go.layout.Shape(
                # type="rect",
                # xref="x",
                # yref="paper",
                # x0="2019-11-25",
                # y0=0,
                # x1="2019-11-26",
                # y1=1,
                # fillcolor="LightSalmon",
                # opacity=0.5,
                # layer="below",
                # line_width=0,
            # )
        # ]
    # )

    fig.update_layout(
        title_text=question,
        legend= {'itemsizing': 'constant'},
    )
    return fig
