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
    data['offer']                    = data['offer'].fillna('Offer B')

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

    return data
