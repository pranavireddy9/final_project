import streamlit as st
import pickle
import numpy as np
import sklearn
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import folium

# Ignore specific warning (replace with your specific warning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
st.set_option('deprecation.showPyplotGlobalUse', False)


data=pd.read_csv("C:/Users/durga prasad/Downloads/classification_data (1).csv",nrows=1000)
data=data.drop_duplicates()

numeric_data = data.select_dtypes(include=['float64', 'int64'])

channelGrouping_dict = {region: index for index, region in enumerate(data['channelGrouping'].unique())}
device_browser_dict = {region: index for index, region in enumerate(data['device_browser'].unique())}
channelGrouping_list = data['channelGrouping'].unique()
device_browser_list = data['device_browser'].unique()


device_operatingSystem_dict = {region: index for index, region in enumerate(data['device_operatingSystem'].unique())}
device_operatingSystem_list = data['device_operatingSystem'].unique()


device_deviceCategory_dict = {region: index for index, region in enumerate(data['device_deviceCategory'].unique())}
geoNetwork_region_dict = {region: index for index, region in enumerate(data['geoNetwork_region'].unique())}
device_deviceCategory_list = data['device_deviceCategory'].unique()
geoNetwork_region_list = data['geoNetwork_region'].unique()


earliest_source_dict = {region: index for index, region in enumerate(data['earliest_source'].unique())}
latest_source_dict = {region: index for index, region in enumerate(data['latest_source'].unique())}
earliest_source_list = data['earliest_source'].unique()
latest_source_list = data['latest_source'].unique()


earliest_medium_dict = {region: index for index, region in enumerate(data['earliest_medium'].unique())}
latest_medium_dict = {region: index for index, region in enumerate(data['latest_medium'].unique())}
earliest_medium_list = data['earliest_medium'].unique()
latest_medium_list = data['latest_medium'].unique()


earliest_keyword_dict = {region: index for index, region in enumerate(data['earliest_keyword'].unique())}
latest_keyword_dict = {region: index for index, region in enumerate(data['latest_keyword'].unique())}
earliest_keyword_list = data['earliest_keyword'].unique()
latest_keyword_list = data['latest_keyword'].unique()

data['product_array'] = data['products_array'].str.split('--').str[0].str.split('////').str[0]
data['product_array'] = data['product_array'].str.lower().str.replace('_', '')
product_array_dict = {region: index for index, region in enumerate(data['product_array'].unique())}





st.set_page_config(layout= "wide")

st.title(":green[**E-commerce Customer Prediction**]")

with st.sidebar:
    option = option_menu('Header', options=["Insights", "PREDICT STATUS"])

if option == "Insights":

    st.title("Exploratory Data Analysis")
    st.subheader("Dataset Overview")
    st.dataframe(data.head())
    st.subheader("Summary Statistics")
    st.table(data.describe())

    # Data Visualization section
    st.title("Data Visualization")

    col1,col2=st.columns(2)

    with col1:

        # Correlation Heatmap
        fig, ax = plt.subplots(figsize=(12, 10))  # Adjust the size as needed

        # Display the correlation matrix
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)


        st.subheader("Pie Chart for 'channelGrouping'")
        channel_counts = data['channelGrouping'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(channel_counts, labels=channel_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title("Pie Chart for 'channelGrouping'")
        st.pyplot()

        # Box plot for 'transactionRevenue'
        st.subheader("Box Plot for 'transactionRevenue'")
        sns.boxplot(x='has_converted', y='transactionRevenue', data=data)
        st.pyplot()

        st.subheader("bar chart for revenue vs region")
        revenue_chart_data = data.groupby('geoNetwork_region')['transactionRevenue'].sum()
        st.bar_chart(revenue_chart_data)

    with col2:
        # Bar plot for 'device_browser'
        st.subheader("Bar Plot for 'device_browser'")
        sns.countplot(x='device_browser', data=data)
        plt.xticks(rotation=45, ha='right')
        st.pyplot()

        st.subheader("Scatter Plot: Transactional Revenue vs. Latitude/Longitude")

        # Create a scatter plot using Matplotlib
        avg_revenue_by_location = data.groupby(['geoNetwork_latitude', 'geoNetwork_longitude'])['transactionRevenue'].mean().reset_index()

        st.subheader("Scatter Plot: Average Transactional Revenue vs. Latitude/Longitude")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='geoNetwork_longitude', y='geoNetwork_latitude', hue='transactionRevenue', size='transactionRevenue', data=avg_revenue_by_location, palette='viridis', sizes=(20, 200))
        plt.title('Average Transactional Revenue vs. Latitude/Longitude')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend(title='Revenue')
        st.pyplot()

        st.subheader("Bar Plot for 'Average Session Time' by 'Conversion Status'")
        plt.figure(figsize=(12, 6))
        sns.barplot(x='has_converted', y='avg_session_time', data=data)
        plt.title("Bar Plot for 'Average Session Time' by 'Conversion Status'")
        plt.xlabel('Conversion Status')
        plt.ylabel('Average Session Time')
        st.pyplot()


















if option == "PREDICT STATUS":

    st.header("PREDICT STATUS (has_converted / not_converted)")
    st.write(" ")

    col1,col2= st.columns(2)
    with col1:
        count_session= st.number_input(label="**Enter the Value for count_session**")
        count_hit= st.number_input(label="**Enter the Value for count_hit**")
        totals_newVisits= st.number_input(label="**Enter the Value for newVisit**/ 0 or 1")
        historic_session=st.number_input(label="**Enter the Value for historic_session**")
        historic_session_page=st.number_input(label="**Enter the Value for historic_session_page**")
        avg_session_time=st.number_input(label="**Enter the Value for avg_session_time**")
        avg_session_time_page=st.number_input(label="**Enter the Value for avg_session_time_page**")
        single_page_rate=st.number_input(label="**Enter the Value for single_page_rate**")
        sessionQualityDim=st.number_input(label="**Enter the Value for sessionQualityDim**")
        visitId_threshold=st.number_input(label="**Enter the Value for visitId_threshold**")
        earliest_visit_id=st.number_input(label="**Enter the Value for earliest_visit_id**")
        earliest_visit_number=st.number_input(label="**Enter the Value for earliest_visit_number**")
        latest_visit_number=st.number_input(label="**Enter the Value for latest_visit_number**")
        time_earliest_visit=st.number_input(label="**Enter the Value for time_earliest_visit**")
        time_latest_visit=st.number_input(label="**Enter the Value for time_latest_visit**")
        avg_visit_time=st.number_input(label="**Enter the Value for avg_visit_time**")
        days_since_first_visit=st.number_input(label="**Enter the Value for days_since_first_visit**")
        visits_per_day=st.number_input(label="**Enter the Value for visits_per_day**")
        bounce_rate=st.number_input(label="**Enter the Value for bounce_rate**")
        num_interactions=st.number_input(label="**Enter the Value for num_interactions**")
        bounces=st.number_input(label="**Enter the Value for bounces**")
        time_on_site=st.number_input(label="**Enter the Value for time_on_site**")
        transactionRevenue=st.number_input(label="**Enter the Value for transactionRevenue**")
        

    with col2:
        channelGrouping= st.selectbox('channelGrouping',options=channelGrouping_list)
        device_browser= st.selectbox('device_browser',options=device_browser_list)
        device_operatingSystem= st.selectbox('device_operatingSystem',options=device_operatingSystem_list)
        device_isMobile= st.text_input(label="**Enter the Value for device_isMobile**/TRUE or FALSE")
        device_deviceCategory= st.selectbox('device_deviceCategory',options=device_deviceCategory_list)
        geoNetwork_region_key= st.selectbox('geoNetwork_region',options=geoNetwork_region_list)
        earliest_medium= st.selectbox('earliest_medium',options=earliest_medium_list)
        latest_medium= st.selectbox('latest_medium',options=latest_medium_list)
        earliest_keyword= st.selectbox('earliest_keyword',options=earliest_keyword_list)
        latest_keyword= st.selectbox('latest_keyword',options=latest_keyword_list)
        earliest_isTrueDirect= st.text_input(label="**Enter the Value for earliest_isTrueDirect**/TRUE or FALSE")
        latest_isTrueDirect= st.text_input(label="**Enter the Value for latest_isTrueDirect**/TRUE or FALSE")
        earliest_source= st.selectbox('earliest_source',options=earliest_source_list)
        latest_source= st.selectbox('latest_source',options=latest_source_list)
        product_array= st.text_input(label="**Enter the Value for product_array**")


    def model_data():
        with open("C:/Users/durga prasad/Desktop/project/.venv/Final Project/Final_model.pkl","rb") as f:
            model=pickle.load(f)
        return model

    # Function to predict
    def predict(model,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac,ad,ae,af,ag,ah,ai):
        pred_value = model.predict([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac,ad,ae,af,ag,ah,ai]])
        return pred_value 

# Create predict button
    if st.button('has convert/not convert'):
        channelGrouping = channelGrouping_dict[channelGrouping]
        device_browser = device_browser_dict[device_browser]
        device_operatingSystem = device_operatingSystem_dict[device_operatingSystem]
        device_deviceCategory = device_deviceCategory_dict[device_deviceCategory]
        geoNetwork_region_fi = geoNetwork_region_dict[geoNetwork_region_key]
        earliest_medium = earliest_medium_dict[earliest_medium]
        latest_medium = latest_medium_dict[latest_medium]
        earliest_keyword = earliest_keyword_dict[earliest_keyword]
        latest_keyword = latest_keyword_dict[latest_keyword]
        earliest_source = earliest_source_dict[earliest_source]
        latest_source = latest_source_dict[latest_source]
        product_array=product_array_dict[product_array]
        totals_newVisits=np.log1p(totals_newVisits)
        geoNetwork_region=np.log1p(geoNetwork_region_fi)
        historic_session_page=np.log1p(historic_session_page)
        visitId_threshold=np.log1p(visitId_threshold)
        earliest_visit_number=np.log1p(earliest_visit_number)
        latest_visit_number=np.log1p(latest_visit_number)
        bounces=np.log1p(bounces)
        time_on_site=np.log1p(time_on_site)
        transactionRevenue=np.log1p(transactionRevenue)
        device_operatingSystem=np.log1p(device_operatingSystem)
        single_page_rate=np.log1p(single_page_rate)
        
        if device_isMobile=='TRUE':
            device_isMobile=1
        else:
            device_isMobile=0

        if earliest_isTrueDirect=='TRUE':
            earliest_isTrueDirect=1
        else:
            earliest_isTrueDirect=0

        if latest_isTrueDirect=='TRUE':
            latest_isTrueDirect=1
        else:
            latest_isTrueDirect=0

        if earliest_medium==latest_medium:

            isMedium = 1
        else:
            isMedium=0
        if earliest_keyword==latest_keyword:
            isKeyword =1
        else:
            isKeyword=0

        if earliest_source==latest_source:
            isSource =1
        else:
            isSource=0

        pred = predict(model_data(), count_session, count_hit, channelGrouping, totals_newVisits,device_browser, device_operatingSystem, device_isMobile,device_deviceCategory, geoNetwork_region, historic_session,historic_session_page, avg_session_time, avg_session_time_page,single_page_rate, sessionQualityDim, visitId_threshold,earliest_visit_id, earliest_visit_number, latest_visit_number,time_earliest_visit, time_latest_visit, avg_visit_time,days_since_first_visit, visits_per_day, bounce_rate,earliest_isTrueDirect, latest_isTrueDirect, num_interactions,bounces, time_on_site, transactionRevenue,product_array, isKeyword, isSource, isMedium)

        if pred[0] == 1:
            st.write("## :green[**Converted to Customer**]")
            st.snow()
        else:
            st.write("## :red[**Not Converted to Customer**]")
