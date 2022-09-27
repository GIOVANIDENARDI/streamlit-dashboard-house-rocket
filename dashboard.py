import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import geopandas
import plotly.express as px
from datetime import datetime

st.set_page_config(layout='wide')


@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    return data


@st.cache(allow_output_mutation=True)
def get_geofile(url):
            geofile = geopandas.read_file(url)
            return geofile


def set_feature(data):
    # add new feature
    data['price_m2'] = data['price'] / (data['sqft_lot'] * 0.09290304)
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    return data


def overview_data(data):
    st.title('Data Overview')
    st.sidebar.title('Location Options')

    f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].sort_values().unique())

    if f_zipcode:
        data = data.loc[data['zipcode'].isin(f_zipcode)]
    else:
        data = data.copy()

    st.dataframe(data)

    c1, c2 = st.columns((1, 1))

    # average metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df1 = pd.merge(m2, df4, on='zipcode', how='inner')
    df1.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQFT LIVING', 'PRICE/M2']

    c1.header('Average Values')
    c1.dataframe(df1, height=600)

    # Descriptive Statistic
    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df2 = pd.concat([min_, max_, media, mediana, std], axis=1).reset_index()
    df2.columns = ['ATTRIBUTES', 'MIN', 'MAX', 'AVG', 'MEDIAN', 'STD']

    c2.header('Descriptive Analysis')
    c2.dataframe(df2, height=600)

    return data


def portfolio_density(data, geofile):
    st.title('Region Overview')

    c1, c2 = st.columns((1, 1))
    c1.header('Portfolio Density')

    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)

    for name, row in data.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup=f'Sold U${row["price"]} on {row["date"]}. '
                            f'Features: {row["sqft_living"]} sqft, '
                            f'{row["bedrooms"]} bedrooms, '
                            f'{row["bathrooms"]} bathrooms, '
                            f'year built {row["yr_built"]}'
                            f'ZipCode: {row["zipcode"]}').add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    c2.header('Price Density')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                labels='ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG PRICE')

    with c2:
        folium_static(region_price_map)
    return None


def commercial_attributes(data):
    st.title('Commercial Attributes')
    st.sidebar.title('Commercial Options')

    # Average Price per Year
    st.header('Average Price per Year Built')

    # Filters
    st.sidebar.subheader('Select Min Year Built')
    min_yr_built = int(data['yr_built'].min())
    max_yr_built = int(data['yr_built'].max())

    # st.write(type(data['yr_built']))
    st.write(type(min_yr_built))
    f_yr_built = st.sidebar.slider('Year Built', min_yr_built, max_yr_built, min_yr_built)

    # Data Selection
    data = data.loc[data['yr_built'] >= f_yr_built]
    df1 = data[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    # Plot
    fig = px.line(df1, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # Average Price per Day
    st.header('Average Price per Day')

    # Filters
    st.sidebar.subheader('Select Min Date')
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Date', min_date, max_date, min_date)

    # Data Selection
    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] >= f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # Histogram
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    # Filter
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)

    data = data.loc[data['price'] <= f_price]

    # Data Plot
    fig = px.histogram(data, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)
    return data


def distribution_properties(data):
    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    # Filters
    f_bedrooms = st.sidebar.selectbox('Min number of bedrooms', data['bedrooms'].sort_values().unique())
    f_bathrooms = st.sidebar.selectbox('Min number of bathrooms', data['bathrooms'].sort_values().unique())
    f_floors = st.sidebar.selectbox('Min number of floors', data['floors'].sort_values().unique())
    f_waterview = st.sidebar.checkbox('Only houses with Water View')

    c1, c2 = st.columns(2)

    # Houses per bedrooms
    c1.header('Houses per bedrooms')
    df = data[data['bedrooms'] >= f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # Houses per bathrooms
    c2.header('Houses per bathrooms')
    df = data[data['bathrooms'] >= f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    # Houses per floors
    c1.header('Houses per floors')
    df = data[data['floors'] >= f_floors]
    fig = px.histogram(df, x='floors', nbins=10)
    c1.plotly_chart(fig, use_container_width=True)

    # Houses per Water View
    c2.header('Houses per Water View')
    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()
    fig = px.histogram(df, x='waterfront', nbins=2)
    c2.plotly_chart(fig, use_container_width=True)
    return None


if __name__ =="__main__":
    # data extraction
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    data = get_data(path)
    geofile = get_geofile(url)

    # transfomation
    data = set_feature(data)
    data = overview_data(data)
    portfolio_density(data, geofile)
    data = commercial_attributes(data)
    distribution_properties(data)
