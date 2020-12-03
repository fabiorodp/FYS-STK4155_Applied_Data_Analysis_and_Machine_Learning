# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import pandas as pd
import numpy as np
import altair as alt
import altair_viewer
import os

# setting parent directory to be accessed
# os.chdir('..')


def candlestick(source, width=800, height=500, view=True):
    """
    Function to generate a interactive candlestick chart for visualization of
    the time series of a financial source.

    Parameters:
    ===================
    :param source: pd.DataFrame: Time series DataFrame containing
                                 OLHCV values.
    :param width: int: The width of the chart.
    :param height: int: The height of the chart.
    :param view: bool: If True, it will return a URL to visualize the chart.

    Return:
    ===================
    The function returns a URL where the interactive chart will be displayed.
    """
    source.reset_index(inplace=True)

    open_close_color = alt.condition("datum.open <= datum.close",
                                     alt.value("#06982d"),
                                     alt.value("#ae1325"))

    base = alt.Chart(source).encode(
        alt.X('DateTime:T',
              axis=alt.Axis(
                  format='%Y/%m/%d',
                  labelAngle=-90,
                  title='Date in 2020')),
        color=open_close_color,
    )

    rule = base.mark_rule().encode(
        alt.Y(
            'low:Q',
            title='Price',
            scale=alt.Scale(zero=False),
        ),
        alt.Y2('high:Q')
    )

    bar = base.mark_bar().encode(
        alt.Y('open:Q'),
        alt.Y2('close:Q')
    )

    chart = (rule + bar).encode(
        tooltip=[alt.Tooltip('DateTime', title='Date'),
                 alt.Tooltip('open', title='Open'),
                 alt.Tooltip('low', title='Low'),
                 alt.Tooltip('high', title='High'),
                 alt.Tooltip('close', title='Close'),
                 alt.Tooltip('volume', title='Volume')]
    ).properties(
        width=width,
        height=height,
        title=f'Candlestick visualization'
    ).interactive()

    if view is True:
        # altair_viewer.show(chart)
        altair_viewer.display(chart)
