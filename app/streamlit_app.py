import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib as plt
import seaborn as sns
import data_preprocessing as dp
import streamlit as st
import plotly.graph_objects as go
import pickle

# Page config and load data ===================================================================================================================
st.set_page_config(layout="wide")
df = pd.read_csv('dataset/telecom_customer_churn.csv')
df_model = pd.read_csv('model/customers_churn_preparated.csv')

# Data Cleaning================================================================================================================================
dp.data_cleaning(df)

# Load Pickle model============================================================================================================================

pickle_load = open('model/classifier.pkl', 'rb')
classifier = pickle.load(pickle_load)

# Function to ML Model Classifier

def prediction(internet_type, contract, dependents, phone_service, internet_service, multiple_lines):
    
    prediction = classifier.predict([[internet_type, contract, dependents, phone_service, internet_service, multiple_lines]])
    print(prediction)
    return prediction

# Interactive filters sidebar ==================================================================================================================

st.sidebar.markdown('# Customer Analysis Indicators')
st.sidebar.divider()
status_churn       = st.sidebar.radio('Customer with an diseabled contract:', ('Yes', 'No'))
type_service       = st.sidebar.radio('Type Service:', ('Internet and Phone', 'Only Internet', 'Only Phone'))
dependents         = st.sidebar.radio('Include Dependets: ', ('Yes', 'No', 'All'))

def convert_df(dataset):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.sidebar.download_button(
    label="Download File",
    data=csv,
    file_name='Customer Analysis Indicators.csv',
    mime='text/csv',
)

st.sidebar.divider()
st.sidebar.markdown("Powered By Erick Vieira")

# Header
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Home', 'Payment Info', 'Products and Services', 'Geographic View', 'Statistic Info', 'Prediction Churn'])

# Payment Info ==================================================================================================================================

with tab1:
    st.header('About Project')
    st.markdown("The 'Customer Analysis Indicators' project is a data solution for a fictitious small/medium-sized Telecom company.") 
    st.markdown("The main objective is to parameterize the KPI's and apply an ML model to determine customer evasion based on their behavior.")
    st.markdown("The company provides various services such as Internet, Telephone, Backup Online and Streaming.")
    st.markdown("In addition to offering different plans, including the possibility of dependents, it offers flexible contracts with prices and payment methods that can be adjusted to the customer's needs. The business model works in B2C, with a diversified service portfolio with services:")
    st.markdown("1 - Domestic and business wired and wireless communication networks with analogue internet (Cable), DSL and fiber optics.")
    st.markdown("2 - Telephone services with multiple lines with short and long distance calling plans.")
    st.markdown("3 - Movie, TV and Music Streaming Services.")
    st.markdown("4 - Device protection service.")
    st.markdown("5 - Online Storage and Backup.")
        
    with st.expander('Dataset Information'):
        st.dataframe(df)

with tab2:
    c0, c01, c02 = st.columns(3)
    with c0:
        if status_churn == 'Yes':
            inf_label = 'Total Churn Customers Selected'
        else:
            inf_label = 'Active customers in the base'

        st.metric(label = inf_label, value = len(df[df['churn'] == status_churn]))

    with c01:
        type_rate = st.selectbox('Select a billing metric:', ('Monthly Charge', 'Total Charge', 'Total Refunds', 'Total Revenue'))
    with c02:
        operation = st.selectbox('Choose the operation to perform:', ('Sum', 'Avarage'))

    c1, c2 = st.columns(2)
    with c1:
        if operation == 'Avarage':
            if type_rate == 'Monthly Charge':
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['monthly_charge', 'churn', 'internet_type']].groupby(['internet_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'monthly_charge', x = 'internet_type', color = 'monthly_charge', text_auto = '.2s', title = 'Average monthly billing rate per internet product ')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['monthly_charge', 'churn', 'internet_type']].groupby(['churn']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'monthly_charge', x = 'internet_type', color = 'monthly_charge', text_auto = '.2s', title = 'Average monthly billing rate per internet product ')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)

            elif type_rate == 'Total Charge':
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['total_charges', 'churn', 'internet_type']].groupby(['internet_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_charges', x = 'internet_type', color = 'total_charges', text_auto = '.2s', title = 'Average total billing rate per internet product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['total_charges', 'churn', 'internet_type']].groupby(['internet_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_charges', x = 'internet_type', color = 'total_charges', text_auto = '.2s', title = 'Average total billing rate per internet product ')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)

            elif type_rate == 'Total Refunds':
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['total_refunds', 'churn', 'internet_type']].groupby(['internet_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_refunds', x = 'internet_type', color = 'total_refunds', text_auto = '.2s', title = 'Average refund rate per internet product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['total_refunds', 'churn', 'internet_type']].groupby(['internet_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_refunds', x = 'internet_type', color = 'total_refunds', text_auto = '.2s', title = 'Average refund rate per internet product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)

            else:
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['total_revenue', 'churn', 'internet_type']].groupby(['internet_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_revenue', x = 'internet_type', color = 'total_revenue', text_auto = '.2s', title = 'Average Revenue Rate by Internet Product ')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['total_revenue', 'churn', 'internet_type']].groupby(['internet_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_revenue', x = 'internet_type', color = 'total_revenue', text_auto = '.2s', title = 'Average Revenue Rate by Internet Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
        else:
            if type_rate == 'Monthly Charge':
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['monthly_charge', 'churn', 'internet_type']].groupby(['internet_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'monthly_charge', x = 'internet_type', color = 'monthly_charge', text_auto = '.2s', title = 'Total monthly billing rate per internet product ')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['monthly_charge', 'churn', 'internet_type']].groupby(['internet_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'monthly_charge', x = 'internet_type', color = 'monthly_charge', text_auto = '.2s', title = 'Total monthly billing rate per internet product ')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)

            elif type_rate == 'Total Charge':
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['total_charges', 'churn', 'internet_type']].groupby(['internet_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_charges', x = 'internet_type', color = 'total_charges', text_auto = '.2s', title = 'Total total billing rate per internet product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['total_charges', 'churn', 'internet_type']].groupby(['internet_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_charges', x = 'internet_type', color = 'total_charges', text_auto = '.2s', title = 'Total total billing rate per internet product ')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)

            elif type_rate == 'Total Refunds':
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['total_refunds', 'churn', 'internet_type']].groupby(['internet_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_refunds', x = 'internet_type', color = 'total_refunds', text_auto = '.2s', title = 'Total refund rate per internet product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['total_refunds', 'churn', 'internet_type']].groupby(['internet_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_refunds', x = 'internet_type', color = 'total_refunds', text_auto = '.2s', title = 'Total refund rate per internet product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)

            else:
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['total_revenue', 'churn', 'internet_type']].groupby(['internet_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_revenue', x = 'internet_type', color = 'total_revenue', text_auto = '.2s', title = 'Total Revenue Rate by Internet Product ')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['total_revenue', 'churn', 'internet_type']].groupby(['internet_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_revenue', x = 'internet_type', color = 'total_revenue', text_auto = '.2s', title = 'Total Revenue Rate by Internet Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)



    with c2:
        if operation == 'Avarage':
            if type_rate == 'Monthly Charge':
                if dependents == 'All':
                    rate_per_customer_type = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_customer_type[['monthly_charge', 'churn', 'customer_type']].groupby(['customer_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'monthly_charge', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Average monthly billing rate by Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['monthly_charge', 'churn', 'customer_type']].groupby(['customer_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'monthly_charge', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Average monthly billing rate per Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)

            elif type_rate == 'Total Charge':
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['total_charges', 'churn', 'customer_type']].groupby(['customer_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_charges', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Average total billing rate per Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['total_charges', 'churn', 'customer_type']].groupby(['customer_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_charges', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Average total billing rate per Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)

            elif type_rate == 'Total Refunds':
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['total_refunds', 'churn', 'customer_type']].groupby(['customer_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_refunds', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Average refund rate per Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['total_refunds', 'churn', 'customer_type']].groupby(['customer_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_refunds', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Average refund rate per Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)

            else:
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['total_revenue', 'churn', 'customer_type']].groupby(['customer_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_revenue', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Average Revenue Rate by Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['total_revenue', 'churn', 'customer_type']].groupby(['customer_type']).mean().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_revenue', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Average Revenue Rate by Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
        else:
            if type_rate == 'Monthly Charge':
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['monthly_charge', 'churn', 'customer_type']].groupby(['customer_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'monthly_charge', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Total monthly billing rate per Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['monthly_charge', 'churn', 'customer_type']].groupby(['customer_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'monthly_charge', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Total monthly billing rate per Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)

            elif type_rate == 'Total Charge':
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['total_charges', 'churn', 'customer_type']].groupby(['customer_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_charges', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Total total billing rate per Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['total_charges', 'churn', 'customer_type']].groupby(['customer_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_charges', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Total total billing rate per Customer Product ')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)

            elif type_rate == 'Total Refunds':
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['total_refunds', 'churn', 'customer_type']].groupby(['customer_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_refunds', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Total refund rate per Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['total_refunds', 'churn', 'customer_type']].groupby(['customer_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_refunds', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Total refund rate per Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)

            else:
                if dependents == 'All':
                    rate_per_product = df[(df['churn'] == status_churn)]
                    m_rate_per_product = rate_per_product[['total_revenue', 'churn', 'customer_type']].groupby(['customer_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_revenue', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Total Revenue Rate by Customer Product ')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)
                else:
                    rate_per_product = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                    m_rate_per_product = rate_per_product[['total_revenue', 'churn', 'customer_type']].groupby(['customer_type']).sum().reset_index()
                    fig_m_rate_per_product = px.bar(m_rate_per_product, y = 'total_revenue', x = 'customer_type', color = 'customer_type', text_auto = '.2s', title = 'Total Revenue Rate by Customer Product')
                    st.plotly_chart(fig_m_rate_per_product, use_container_width = True)
                    with st.expander('More Info'):
                        st.dataframe(m_rate_per_product)

    c3, c4, c5 = st.columns(3)
    with c3:
        if dependents == 'All':
            payment = df[(df['churn'] == status_churn) & (df['product_type'] == type_service)]
            payment_method = payment.groupby(['payment_method'])['churn'].count().reset_index()
            fig = px.bar(payment_method, x = 'payment_method', y = 'churn', color = 'payment_method', text_auto='.2s', title="Payment Method more used by customers")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("More Info"):
                st.dataframe(payment_method)
        else:
            payment = df[(df['churn'] == status_churn) & (df['product_type'] == type_service) & (df['dependents'] == dependents)]
            payment_method = payment.groupby(['payment_method'])['churn'].count().reset_index()
            fig = px.bar(payment_method, x = 'payment_method', y = 'churn', color = 'payment_method', text_auto='.2s', title="Payment Method more used by customers")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("More Info"):
                st.dataframe(payment_method)
    with c4:
        if dependents == 'All':
            contract = df[(df['churn'] == status_churn) & (df['product_type'] == type_service)]
            contract_type = contract.groupby(['contract'])['churn'].count().reset_index()
            fig_contract = px.bar(contract_type, x = 'contract', y = 'churn', color = 'contract', text_auto='.2s', title="Contract Type by customers")
            st.plotly_chart(fig_contract, use_container_width=True)
            with st.expander("More Info"):
                st.dataframe(contract_type)
        else:
            contract = df[(df['churn'] == status_churn) & (df['product_type'] == type_service) & (df['dependents'] == dependents)]
            contract_type = contract.groupby(['contract'])['churn'].count().reset_index()
            fig_contract = px.bar(contract_type, x = 'contract', y = 'churn', color = 'contract', text_auto='.2s', title="Contract Type by customers")
            st.plotly_chart(fig_contract, use_container_width=True)
            with st.expander("More Info"):
                st.dataframe(contract_type)

    with c5:
        if dependents == 'All':
            paper = df[(df['churn'] == status_churn) & (df['product_type'] == type_service)]
            paperless = paper.groupby(['billing_type'])['churn'].count().reset_index()
            fig = px.bar(paperless, x = 'billing_type', y = 'churn', color = 'billing_type', text_auto='.2s', title="Billing type used by customers")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("More Info"):
                st.dataframe(paperless)
        else:
            paper = df[(df['churn'] == status_churn) & (df['product_type'] == type_service) & (df['dependents'] == dependents)]
            paperless = paper.groupby(['billing_type'])['churn'].count().reset_index()
            fig = px.bar(paperless, x = 'billing_type', y = 'churn', color = 'billing_type', text_auto='.2s', title="Billing type used by customers")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("More Info"):
                st.dataframe(paperless)


# Products and Services ==========================================================================================================================
with tab3:

    if status_churn == 'Yes':
        inf_label = 'Total Churn Customers Selected'
    else:
        inf_label = 'Active customers in the base'

    st.metric(label = inf_label, value = len(df[df['churn'] == status_churn]))
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            if dependents == 'All':
                d_phone_type = df[df['churn'] == status_churn]
                phone_type = d_phone_type.groupby(['phone_service', 'multiple_lines'])['churn'].count().reset_index()
                fig_phone = px.bar(phone_type, x = 'phone_service', y = 'churn', color = 'multiple_lines', text_auto = '.2s', title = 'Quantity of Customers by Phone Service')
                st.plotly_chart(fig_phone, use_container_width = True)
                with st.expander('More Info'):
                    st.dataframe(phone_type)
            else:
                d_phone_type = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                phone_type = d_phone_type.groupby(['phone_service', 'multiple_lines'])['churn'].count().reset_index()
                fig_phone = px.bar(phone_type, x = 'phone_service', y = 'churn', color = 'multiple_lines', text_auto = '.2s', title = 'Quantity of Customers by Phone Service')
                st.plotly_chart(fig_phone, use_container_width = True)
                with st.expander('More Info'):
                    st.dataframe(phone_type)

        with col2:
            if dependents == 'All':
                int_type = df[(df['churn'] == status_churn) & (df['product_type'] == type_service)]
                multi_services = int_type.groupby(['internet_type'])['churn'].count().reset_index()
                fig_multi_services = px.bar(multi_services, x = 'internet_type', y = 'churn', color = 'internet_type', text_auto = '.2s', title = 'Quantity of Customers by internet Type')
                st.plotly_chart(fig_multi_services, use_container_width = True)
                with st.expander('More Info'):
                    st.dataframe(multi_services) 
            else:
                int_type = df[(df['churn'] == status_churn) & (df['product_type'] == type_service) & (df['dependents'] == dependents)]
                multi_services = int_type.groupby(['internet_type'])['churn'].count().reset_index()
                fig_multi_services = px.bar(multi_services, x = 'internet_type', y = 'churn', color = 'internet_type', text_auto = '.2s', title = 'Quantity of Customers by internet Type')
                st.plotly_chart(fig_multi_services, use_container_width = True)
                with st.expander('More Info'):
                    st.dataframe(multi_services)  
        with col3:
            d_type = df[df['churn'] == 'No']
            dep = d_type.groupby(['dependents'])['churn'].count().reset_index()
            fig_p_type = px.bar(dep, x = 'dependents', y = 'churn', color = 'dependents', text_auto = '.2s', title = 'Total of Dependents in base')
            st.plotly_chart(fig_p_type, use_container_width = True)
            with st.expander('More Info'):
                st.dataframe(dep)

        col4, col5 = st.columns(2)
        with col4:
            if dependents == 'All':
                p_type = df[(df['churn'] == status_churn)]
                product_type = p_type.groupby(['product_type'])['churn'].count().reset_index()
                fig_p_type = px.bar(product_type, x = 'product_type', y = 'churn', color = 'product_type', text_auto = '.2s', title = 'Customers by Product')
                st.plotly_chart(fig_p_type, use_container_width = True)
                with st.expander('More Info'):
                    st.dataframe(product_type)
            else:
                p_type = df[(df['churn'] == status_churn) & (df['dependents'] == dependents)]
                product_type = p_type.groupby(['product_type'])['churn'].count().reset_index()
                fig_p_type = px.bar(product_type, x = 'product_type', y = 'churn', color = 'product_type', text_auto = '.2s', title = 'Customers by Product')
                st.plotly_chart(fig_p_type, use_container_width = True)
                with st.expander('More Info'):
                    st.dataframe(product_type)    

        with col5:
            if dependents == 'All':
                c_type = df[(df['churn'] == status_churn) & (df['product_type'] == type_service)]
                customer_type = c_type.groupby(['customer_type'])['churn'].count().reset_index()
                fig_customer_type = px.bar(customer_type, x = 'customer_type', y = 'churn', color = 'customer_type', text_auto = '.2s', title = 'Customers by Services')
                st.plotly_chart(fig_customer_type, use_container_width = True)
                with st.expander('More Info'):
                    st.dataframe(customer_type)
            else:
                c_type = df[(df['churn'] == status_churn) & (df['product_type'] == type_service) & (df['dependents'] == dependents)]
                customer_type = c_type.groupby(['customer_type'])['churn'].count().reset_index()
                fig_customer_type = px.bar(customer_type, x = 'customer_type', y = 'churn', color = 'customer_type', text_auto = '.2s', title = 'Customers by Services')
                st.plotly_chart(fig_customer_type, use_container_width = True)
                with st.expander('More Info'):
                    st.dataframe(customer_type)

# Geographic View ================================================================================================================================

with tab4:
    cols1, cols2 = st.columns(2)
    with cols1:
        product = st.multiselect('Select some Internet Product:', df['internet_type'].unique())
    with cols2:
        churn = st.selectbox('Select a status:', ['All', 'Churned', 'Not Churned'])

    if len(product) > 0:
        df_cities1 = df[df['internet_type'].isin(product)]
        if churn == 'Churned':
            df_cities2 = df_cities1[df_cities1['churn'] == 'Yes']
            df_map = df_cities2[['churn', 'latitude', 'longitude', 'city', 'internet_type']]
            df_map['point_size'] = df_map['churn'].apply(lambda x: 1 if x == 'No' else 4)
            df_map = df_map.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

            mapa = px.scatter_mapbox(df_map, lat='lat', hover_name ='city', hover_data= ['churn', 'internet_type'] , lon='lon', size = 'point_size', zoom=5, color_discrete_sequence=['#0598ad'], height=600)

            mapa.update_layout(mapbox_style='open-street-map')
            mapa.update_layout(height=500, margin = {'t': 0, 'r':0, 'b': 0, 'l':0})

            st.plotly_chart(mapa, use_container_width=True)

        elif churn == 'Not Churned':
            df_cities2 = df_cities1[df_cities1['churn'] == 'No']
            df_map = df_cities2[['churn', 'latitude', 'longitude', 'city', 'internet_type']]
            df_map['point_size'] = df_map['churn'].apply(lambda x: 1 if x == 'No' else 4)
            df_map = df_map.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

            mapa = px.scatter_mapbox(df_map, lat='lat', hover_name ='city', hover_data= ['churn', 'internet_type'] , lon='lon', size = 'point_size', zoom=5, color_discrete_sequence=['#0598ad'], height=600)

            mapa.update_layout(mapbox_style='open-street-map')
            mapa.update_layout(height=500, margin = {'t': 0, 'r':0, 'b': 0, 'l':0})

            st.plotly_chart(mapa, use_container_width=True)
        else:
            df_map = df_cities1[['churn', 'latitude', 'longitude', 'city', 'internet_type']]
            df_map['point_size'] = df_map['churn'].apply(lambda x: 1 if x == 'No' else 4)
            df_map = df_map.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

            mapa = px.scatter_mapbox(df_map, lat='lat', hover_name ='city', hover_data= ['churn', 'internet_type'] , lon='lon', size = 'point_size', zoom=5, color_discrete_sequence=['#0598ad'], height=600)

            mapa.update_layout(mapbox_style='open-street-map')
            mapa.update_layout(height=500, margin = {'t': 0, 'r':0, 'b': 0, 'l':0})

            st.plotly_chart(mapa, use_container_width=True)

    else:
        if churn == 'Churned':
            df_cities2 = df[df['churn'] == 'Yes']
            df_map = df_cities2[['churn', 'latitude', 'longitude', 'city', 'internet_type']]
            df_map['point_size'] = df_map['churn'].apply(lambda x: 1 if x == 'No' else 4)
            df_map = df_map.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

            mapa = px.scatter_mapbox(df_map, lat='lat', hover_name ='city', hover_data= ['churn', 'internet_type'] , lon='lon', size = 'point_size', zoom=5, color_discrete_sequence=['#0598ad'], height=600)

            mapa.update_layout(mapbox_style='open-street-map')
            mapa.update_layout(height=500, margin = {'t': 0, 'r':0, 'b': 0, 'l':0})

            st.plotly_chart(mapa, use_container_width=True)

        elif churn == 'Not Churned':
            df_cities2 = df[df['churn'] == 'No']
            df_map = df_cities2[['churn', 'latitude', 'longitude', 'city', 'internet_type']]
            df_map['point_size'] = df_map['churn'].apply(lambda x: 1 if x == 'No' else 4)
            df_map = df_map.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

            mapa = px.scatter_mapbox(df_map, lat='lat', hover_name ='city', hover_data= ['churn', 'internet_type'] , lon='lon', size = 'point_size', zoom=5, color_discrete_sequence=['#0598ad'], height=600)

            mapa.update_layout(mapbox_style='open-street-map')
            mapa.update_layout(height=500, margin = {'t': 0, 'r':0, 'b': 0, 'l':0})

            st.plotly_chart(mapa, use_container_width=True)
        else:
            df_map = df[['churn', 'latitude', 'longitude', 'city', 'internet_type']]
            df_map['point_size'] = df_map['churn'].apply(lambda x: 1 if x == 'No' else 4)
            df_map = df_map.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

            mapa = px.scatter_mapbox(df_map, lat='lat', hover_name ='city', hover_data= ['churn', 'internet_type'] , lon='lon', size = 'point_size', zoom=5, color_discrete_sequence=['#0598ad'], height=600)

            mapa.update_layout(mapbox_style='open-street-map')
            mapa.update_layout(height=500, margin = {'t': 0, 'r':0, 'b': 0, 'l':0})

            st.plotly_chart(mapa, use_container_width=True)

# statistic info ================================================================================================================================

with tab5:
    df_num = df.select_dtypes(exclude = 'object')
    df_num = df_num[['age', 'numberof_referrals',
       'tenurein_months', 'avg_monthly_long_distance_charges',
       'avg_monthly_gb_download', 'monthly_charge', 'total_charges',
       'total_refunds', 'total_extra_data_charges',
       'total_long_distance_charges', 'total_revenue']]
    
    central_tendency1 = pd.DataFrame(df_num.apply(lambda x: np.mean(x))).T
    central_tendency2 = pd.DataFrame(df_num.apply(lambda x: np.median(x))).T

    dispersion1 = pd.DataFrame(df_num.apply(lambda x: np.std(x))).T
    dispersion2 = pd.DataFrame(df_num.apply(max)).T
    dispersion3 = pd.DataFrame(df_num.apply(min)).T
    dispersion4 = pd.DataFrame(df_num.apply(lambda x: x.min() - x.max())).T
    dispersion5 = pd.DataFrame(df_num.apply(lambda x: x.skew())).T
    dispersion6 = pd.DataFrame(df_num.apply(lambda x: x.kurtosis())).T

    metrics = pd.concat([central_tendency1, central_tendency2, dispersion1, dispersion2, dispersion3, dispersion4, dispersion5, dispersion6]).T.reset_index()
    metrics.columns = ['Attributes', 'Avarage', 'Median', 'Standart Deviation', 'Max', 'Min', 'Range', 'Skew', 'Kurtosis']

    st.dataframe(metrics)

# Prediction ML Model ================================================================================================================================

with tab6:
    columns_input1, columns_input2 = st.columns(2)
    with columns_input1:        
        internet_service = st.selectbox('Have an Internet Service? No = 0 | Yes = 0', df_model['internet_service'].unique())
        internet_type = st.selectbox('Select an Internet Type: Cable - 1 | DLS - 2 | Fiber - 3' , df_model['internet_type'].unique())
        dependents = st.selectbox('Have Dependents? No = 0 | Yes = 0', df_model['dependents'].unique())
        

    with columns_input2:
        phone_service = st.selectbox('Have a Phone Service? No = 0 | Yes = 0', df_model['phone_service'].unique())
        multiple_lines = st.selectbox('Have a Multiple Lines? No = 0 | Yes = 0', df_model['multiple_lines'].unique())
        contract = st.selectbox('Select a Contract Type: Month to Month - 1 | One Year- 2 | Two Years - 3', df_model['contract'].unique())

    if st.button('Predict'):
        result = prediction(internet_type, contract, dependents, phone_service, internet_service, multiple_lines)
        if internet_service == 0:
            internet_service_text = 'No Internet Service'
        else:
            internet_service_text = 'With Internet Service'

        if dependents == 0:
            dependents_text = 'No Dependents'
        else:
            dependents_text = 'With dependents' 

        if phone_service == 0:
            phone_service_text = 'No Phone service'
        else:
            phone_service_text = 'With Phone service'    

        if multiple_lines == 0:
            multiple_lines_text = 'No Multiple Lines'
        else:
            multiple_lines_text = 'With Multiple Lines' 

        if contract == 0:
            contract_text = 'Month to Month Contract'
        elif contract == 1:
            contract_text = 'One Year Contract'   
        else:
            contract_text = 'Two Years Contract'

        if internet_type == 0:
            internet_type_text = 'Internet Analogic Cable'
        elif internet_type == 1:
            internet_type_text = 'Internet DSL'   
        else:
            internet_type_text = 'Internet Fiber Optic'

        if result == 0:
            st.success("This customer's behavior doesn't demonstrate a potential churn pattern:")
            st.success([internet_service_text, contract_text, dependents_text, phone_service_text, internet_type_text, multiple_lines_text])
        else:
            st.warning("This customer's behavior is evaluated as potential churn pattern:")
            st.warning([internet_service_text, contract_text, dependents_text, phone_service_text, internet_type_text, multiple_lines_text])
