import pandas as pd
import numpy as np
import inflection
import math

def data_cleaning(data):
    #Removing spaces bettween strings
    columns = data.columns
    columns_strip = data.columns.str.strip().str.replace(' ', '')
    columns_strip

    #Applying snakecase patterns
    snakecase = lambda x: inflection.underscore(x)
    new_columns = list(map(snakecase, columns_strip))
    data.columns = new_columns

    # Using lambda function for fill all NAN data with the data that most repeated 
    data['avg_monthly_long_distance_charges'] = data['avg_monthly_long_distance_charges'].apply(lambda x: 18.26 if math.isnan(x) else (x))

    # Using fillna function for fill all NAN data with the data that most repeated 
    data['multiple_lines']           = data['multiple_lines'].fillna('No')
    data['internet_type']            = data['internet_type'].fillna('Fiber Optic')
    data['avg_monthly_gb_download']  = data['avg_monthly_gb_download'].fillna(19.0)
    data['online_security']          = data['online_security'].fillna('No')
    data['online_backup']            = data['online_backup'].fillna('No')
    data['device_protection_plan']   = data['device_protection_plan'].fillna('No')
    data['premium_tech_support']     = data['premium_tech_support'].fillna('No')
    data['streaming_tv']             = data['streaming_tv'].fillna('No')
    data['streaming_movies']         = data['streaming_movies'].fillna('No')
    data['streaming_music']          = data['streaming_music'].fillna('No')
    data['unlimited_data']           = data['unlimited_data'].fillna('Yes')

    # Drop the columns that have 73% of NA data
    data.drop(['churn_category', 'churn_reason'], axis = 1, inplace = True)

    data.rename({'customer_status': 'churn'}, axis=1, inplace = True)
    data['churn'] = data['churn'].apply(lambda x: 'Yes' if x == 'Churned' else 'No')

    data.rename(columns = {'numberof_dependents' : 'dependents', 'paperless_billing':'billing_type'}, inplace = True)
    data['dependents'] = data['dependents'].astype(str)
    data['dependents'] = data['dependents'].apply(lambda x: 'No' if x == '0' else 'Yes')

    data['monthly_charge'] = data['monthly_charge'].apply(lambda x: 0 if x <= 0 else x)

    data['billing_type'] = data['billing_type'].apply(lambda x: 'Digital' if x == 'Yes' else 'Paper')

    data['customer_type'] = 'standart'

    data.loc[(data['online_security'] == 'Yes') & (data['online_backup'] == 'Yes') 
            & (data['device_protection_plan'] == 'Yes') & (data['streaming_tv'] == 'Yes') 
            & (data['streaming_movies'] == 'Yes') & (data['streaming_music'] == 'Yes')
            & (data['unlimited_data'] == 'Yes'), 'customer_type'] = 'premium'

    data.loc[(data['online_security'] == 'No') & (data['online_backup'] == 'No') 
            & (data['device_protection_plan'] == 'No') & (data['streaming_tv'] == 'Yes') 
            & (data['streaming_movies'] == 'Yes') & (data['streaming_music'] == 'Yes')
            & (data['unlimited_data'] == 'No'), 'customer_type'] = 'only streaming'

    data.loc[(data['online_security'] == 'Yes') & (data['online_backup'] == 'Yes') 
            & (data['device_protection_plan'] == 'Yes') & (data['streaming_tv'] == 'No') 
            & (data['streaming_movies'] == 'No') & (data['streaming_music'] == 'No')
            & (data['unlimited_data'] == 'Yes'), 'customer_type'] = 'only data service'
    
    data['product_type'] = 'Internet and Phone'
    data.loc[(data['internet_service'] == 'Yes') & (data['phone_service'] == 'No'), 'product_type'] = 'Only Internet'
    data.loc[(data['internet_service'] == 'No') & (data['phone_service'] == 'Yes'), 'product_type'] = 'Only Phone'


def data_cleaning_model(dataset):
    #Removing spaces bettween strings
    columns = dataset.columns
    columns_strip = dataset.columns.str.strip().str.replace(' ', '')
    columns_strip

    #Applying snakecase patterns
    snakecase = lambda x: inflection.underscore(x)
    new_columns = list(map(snakecase, columns_strip))
    dataset.columns = new_columns

    # Using lambda function for fill all NAN data with the data that most repeated 
    dataset['avg_monthly_long_distance_charges'] = dataset['avg_monthly_long_distance_charges'].apply(lambda x: 18.26 if math.isnan(x) else (x))

    # Using fillna function for fill all NAN data with the data that most repeated 
    dataset['multiple_lines']           = dataset['multiple_lines'].fillna('No')
    dataset['internet_type']            = dataset['internet_type'].fillna('Fiber Optic')
    dataset['avg_monthly_gb_download']  = dataset['avg_monthly_gb_download'].fillna(19.0)
    dataset['online_security']          = dataset['online_security'].fillna('No')
    dataset['online_backup']            = dataset['online_backup'].fillna('No')
    dataset['device_protection_plan']   = dataset['device_protection_plan'].fillna('No')
    dataset['premium_tech_support']     = dataset['premium_tech_support'].fillna('No')
    dataset['streaming_tv']             = dataset['streaming_tv'].fillna('No')
    dataset['streaming_movies']         = dataset['streaming_movies'].fillna('No')
    dataset['streaming_music']          = dataset['streaming_music'].fillna('No')
    dataset['unlimited_data']           = dataset['unlimited_data'].fillna('Yes')

    # Drop the columns that have 73% of NA data
    dataset.drop(['churn_category', 'churn_reason'], axis = 1, inplace = True)

    dataset.rename({'customer_status': 'churn'}, axis=1, inplace = True)
    dataset['churn'] = dataset['churn'].apply(lambda x: 'Yes' if x == 'Churned' else 'No')

    dataset.rename(columns = {'numberof_dependents' : 'dependents'}, inplace = True)
    dataset['dependents'] = dataset['dependents'].astype(str)
    dataset['dependents'] = dataset['dependents'].apply(lambda x: 'No' if x == '0' else 'Yes')

    dataset['monthly_charge'] = dataset['monthly_charge'].apply(lambda x: 0 if x <= 0 else x)

    dataset['customer_type'] = 'standart'

    dataset.loc[(dataset['online_security'] == 'Yes') & (dataset['online_backup'] == 'Yes') 
            & (dataset['device_protection_plan'] == 'Yes') & (dataset['streaming_tv'] == 'Yes') 
            & (dataset['streaming_movies'] == 'Yes') & (dataset['streaming_music'] == 'Yes')
            & (dataset['unlimited_data'] == 'Yes'), 'customer_type'] = 'premium'

    dataset.loc[(dataset['online_security'] == 'No') & (dataset['online_backup'] == 'No') 
            & (dataset['device_protection_plan'] == 'No') & (dataset['streaming_tv'] == 'Yes') 
            & (dataset['streaming_movies'] == 'Yes') & (dataset['streaming_music'] == 'Yes')
            & (dataset['unlimited_data'] == 'No'), 'customer_type'] = 'only streaming'

    dataset.loc[(dataset['online_security'] == 'Yes') & (dataset['online_backup'] == 'Yes') 
            & (dataset['device_protection_plan'] == 'Yes') & (dataset['streaming_tv'] == 'No') 
            & (dataset['streaming_movies'] == 'No') & (dataset['streaming_music'] == 'No')
            & (dataset['unlimited_data'] == 'Yes'), 'customer_type'] = 'only data service'
    
    dataset['product_type'] = 'Internet and Phone'
    dataset.loc[(dataset['internet_service'] == 'Yes') & (dataset['phone_service'] == 'No'), 'product_type'] = 'Only Internet'
    dataset.loc[(dataset['internet_service'] == 'No') & (dataset['phone_service'] == 'Yes'), 'product_type'] = 'Only Phone'


