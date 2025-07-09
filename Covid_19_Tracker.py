#!/usr/bin/env python
# coding: utf-8

# # 🌍 COVID-19 & Demographic Insights Dashboard
# 
# Welcome to this interactive Jupyter Notebook presenting a series of visual dashboards and data analyses based on the latest available COVID-19 and population datasets. The notebook combines statistical insights and interactive visualizations built using **Plotly** and **Dash** to explore global health and demographic trends.
# 
# ## 📌 What you'll find inside:
# 
# - **COVID-19 Map – Positive Cases per Capita**  
#   An interactive map showing the number of positive COVID-19 cases per capita across different continents and countries.
# 
# - **Dashboard – Total Cases, Deaths, Tests, Vaccinations, and Vaccinated People**  
#   A flexible dashboard that lets you explore different COVID-19 metrics by continent and country using map-based markers.
# 
# - **Top Countries by Total Vaccinations and Vaccinations per Capita**  
#   Dual bar charts comparing countries with the highest absolute and relative vaccination coverage.
# 
# - **COVID-19 Comparison – Czech Republic vs. Slovakia**  
#   A focused comparison of COVID-19 developments between CZ and SK based on selected metrics.
# 
# - **Top 10 Countries by Population**  
#   A simple yet powerful chart showing which countries have the largest populations globally.
# 
# - **Relationship Between Population and Life Expectancy**  
#   A scatter plot examining the link between a country’s population size and its average life expectancy.
# 
# ---
# Let's dive in! 🔎
# 

# In[ ]:


get_ipython().system('pip install --upgrade plotly')
get_ipython().system('pip install --upgrade seaborn')
get_ipython().system('pip install --upgrade pandas')
get_ipython().system('pip install --upgrade dash')

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/project_1_python.csv')


# In[ ]:


df.head()


# In[ ]:


top_10_countries = df.groupby('location')['population'].max().nlargest(10)
top_10_countries


# ## Global Population Rankings – Top 10 Countries

# In[ ]:


plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

sns.barplot(x=top_10_countries.index, y=top_10_countries.values)

plt.xticks(rotation=45, ha='right')

plt.xlabel("Country")
plt.ylabel("Population in Bilions")
plt.title("Top 10 Countries by Population")

plt.show()


# ## Correlation Between Population and Life Expectancy

# In[ ]:


fig = px.scatter(df,
                 x="population",
                 y="life_expectancy",
                 color="continent",
                 color_discrete_sequence=px.colors.qualitative.Pastel,
                 title="Relationship between Population and Life Expectancy",
                 log_x=True,
                 hover_name="location")

fig.show()


# ## Comparative COVID-19 Dashboard: CZ vs. SK

# In[ ]:


# Filtered data for Czech Republic and Slovakia
df_filtered = df[(df['location'] == 'Czechia') | (df['location'] == 'Slovakia')]

fig = px.line(df_filtered,
              x='date',
              y='new_cases',
              color='location',
              color_discrete_sequence=['#636EFA', '#EF553B'],  # Custom color palette
              title='New COVID-19 Cases over Time: Czech Republic vs. Slovakia',
              markers=True)

fig.update_layout(xaxis_title='Date', yaxis_title='New Cases')

fig.show()


# ## COVID-19 – Cases per Capita (Map View)

# In[60]:


import plotly.express as px

df_map = df[df['date'] == df['date'].max()].copy()
df_map['ratio'] = (df_map['total_cases'] / df_map['population'] * 100).round(2)
df_map['cases_per_person'] = df_map['total_cases'] / df_map['population']

cases_map = px.scatter_map(data_frame=df_map,
                           lat='latitude', lon='longitude',
                           color='continent',
                           size='cases_per_person',
                           size_max=20,
                           hover_data={
                               'location': True,
                               'total_cases': True,
                               'continent': False,
                               'cases_per_person': False,
                               'ratio': True,
                               'latitude': False,
                               'longitude': False,
                           },
                           zoom=1,
                           map_style='open-street-map',
                           title='COVID-19 map - Positive cases per person')

cases_map.show()


# ## Dashboard – Cumulative Number of Positive Cases in Selected Country
# 

# In[61]:


df['date'] = pd.to_datetime(df['date'])

app = dash.Dash(__name__)
app.title = "COVID-19 Dashboard"

available_countries = sorted(df['location'].dropna().unique())

app.layout = html.Div([
    html.H1(id='dashboard-title', style={'textAlign': 'center'}),

    html.Label("Select a country:", style={'fontWeight': 'bold'}),
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': country, 'value': country} for country in available_countries],
        value='Czechia',
        style={'width': '50%'}
    ),

    html.Div([
        dcc.Graph(id='cases-graph'),
        dcc.Graph(id='deaths-graph')
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'})
])

@app.callback(
    [Output('cases-graph', 'figure'),
     Output('deaths-graph', 'figure'),
     Output('dashboard-title', 'children')],
    [Input('country-dropdown', 'value')]
)
def update_dashboard(selected_country):
    filtered_df = df[df['location'] == selected_country]

    fig_cases = px.line(filtered_df, x='date', y='total_cases',
                        title='Cumulative COVID-19 Cases',
                        labels={'total_cases': 'Total Cases', 'date': 'Date'})

    fig_deaths = px.line(filtered_df, x='date', y='total_deaths',
                         title='Cumulative COVID-19 Deaths',
                         labels={'total_deaths': 'Total Deaths', 'date': 'Date'})

    title = f"Cumulative number of positive cases in {selected_country}"
    return fig_cases, fig_deaths, title

app.run(mode='inline')


# ## Dashboard – Total Cases, Deaths, Tests, Vaccinations, and Vaccinated People
# 

# In[67]:


data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/project_1_python.csv')

data = data.rename(columns={
    'location': 'country',
    'latitude': 'lat',
    'longitude': 'lon',
    'people_vaccinated': 'total_vaccinated'
})

app = dash.Dash(__name__)

continents = data['continent'].dropna().unique()
metrics = {
    'total_cases': 'Total Cases',
    'total_deaths': 'Total Deaths',
    'total_tests': 'Total Tests',
    'total_vaccinations': 'Total Vaccinations',
    'total_vaccinated': 'Total Vaccinated People'
}

app.layout = html.Div([
    html.H1(id='dashboard-title', style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Continent:"),
        dcc.Dropdown(
            id='continent-filter',
            options=[{'label': cont, 'value': cont} for cont in continents],
            value='Europe'
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Select Metric:"),
        dcc.Dropdown(
            id='metric-filter',
            options=[{'label': label, 'value': key} for key, label in metrics.items()],
            value='total_cases'
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    dcc.Graph(id='map-visualization')
])

@app.callback(
    [Output('map-visualization', 'figure'),
     Output('dashboard-title', 'children')],
    [Input('continent-filter', 'value'),
     Input('metric-filter', 'value')]
)
def update_map(continent, metric):
    filtered_data = data[data['continent'] == continent]
    filtered_data = filtered_data.dropna(subset=["lat", "lon", metric])

    fig = px.scatter_mapbox(
        filtered_data,
        lat="lat",
        lon="lon",
        size=metric,
        hover_name="country",
        hover_data={metric: True, "lat": True, "lon": True},
        zoom=2,
        size_max=40
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    title = f"COVID-19 - {metrics[metric]} in {continent}"
    return fig, title

if __name__ == '__main__':
    app.run(debug=True)


# 

# ## COVID-19 Vaccination Leaders – Total vs. Per Capita
# 

# In[69]:


data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/project_1_python.csv')
data = data.rename(columns={
    'location': 'country',
    'people_vaccinated': 'total_vaccinated'
})

# Group by country and aggregate latest data
data = data.groupby('country', as_index=False).agg({
    'continent': 'first',
    'population': 'first',
    'total_vaccinated': 'max'
})

# Drop rows with missing necessary values
data = data.dropna(subset=['total_vaccinated', 'population'])

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("COVID-19 Vaccination Dashboard", style={"textAlign": "center"}),

    html.Label("Number of countries to display:"),
    dcc.Slider(
        id='top-n-slider',
        min=5,
        max=20,
        step=5,
        marks={i: str(i) for i in range(5, 21, 5)},
        value=5
    ),

    html.Div([
        dcc.Graph(id='top-vaccinated-countries'),
        dcc.Graph(id='top-ratio-vaccinated-countries')
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),

    html.P("Note: Vaccination values can exceed population due to multiple doses per person (e.g., boosters).",
           style={"textAlign": "center", "fontStyle": "italic", "marginTop": "20px"})
])

@app.callback(
    [Output('top-vaccinated-countries', 'figure'),
     Output('top-ratio-vaccinated-countries', 'figure')],
    [Input('top-n-slider', 'value')]
)
def update_charts(n):
    filtered = data[pd.to_numeric(data['total_vaccinated'], errors='coerce') > 0].copy()

    top_vacc = filtered.sort_values('total_vaccinated', ascending=False).head(n)

    filtered['vaccination_ratio'] = filtered['total_vaccinated'] / filtered['population']
    top_ratio = filtered.sort_values('vaccination_ratio', ascending=False).head(n)

    fig_vacc = px.bar(
        top_vacc,
        x='country',
        y='total_vaccinated',
        title=f'Top {n} Countries by Total Vaccinations',
        labels={'total_vaccinated': 'Total Vaccinated People'}
    )

    fig_ratio = px.bar(
        top_ratio,
        x='country',
        y='vaccination_ratio',
        title=f'Top {n} Countries by Vaccination per Capita',
        labels={'vaccination_ratio': 'Vaccinated / Population'}
    )

    return fig_vacc, fig_ratio

if __name__ == '__main__':
    app.run(debug=True)

