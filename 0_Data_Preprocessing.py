# 1
# Getting Required Packages and tools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array
from sklearn.preprocessing import LabelEncoder

# 2
# Uploading the data and assigning Variable
Original_Data = pd.read_csv("investments_VC_Copy.csv", encoding="ISO-8859-1")

# Copy of Data
copy_data = Original_Data

# 3
# Extracting Sample Points
# Dropping missing value threshold
threshold = 0.7
# Dropping columns with missing value rate higher than threshold
copy_data = copy_data[copy_data.columns[copy_data.isnull().mean() < threshold]]
# Dropping rows with missing value rate higher than threshold
# copy_data = copy_data.loc[copy_data.isnull().mean(axis=1) < threshold]

# Dropping Relevant attributes
copy_data = copy_data.drop(columns=['name', 'funding_total_usd', 'category_list', 'city', 'founded_at',
                                    'founded_month', 'founded_quarter', 'founded_year'])
# NOTE: Deleting 'funding_total_usd', because dtype is mixed and very noisy

# First_last_funding_dates
first_last_funding_dates = copy_data.iloc[:, 4:6]
copy_data = copy_data.drop(columns=['first_funding_at', 'last_funding_at'])

# Creating back 'funding_total_usd' by additions of Relevant attributes
copy_data['funding_total_usd'] = copy_data.iloc[:, -21:-1].sum(axis=1)

# Working on Outliers
# Checking Outliers w.r.t to Standard deviations for funding_total_usd
factor = 3
upper_lim = copy_data['funding_total_usd'].mean() + copy_data['funding_total_usd'].std() * factor
lower_lim = copy_data['funding_total_usd'].mean() - copy_data['funding_total_usd'].std() * factor
copy_data = copy_data[(copy_data['funding_total_usd'] < upper_lim) & (copy_data['funding_total_usd'] > lower_lim)]

# Checking Outliers w.r.t to Percentiles for funding_total_usd
# upper_lim = copy_data['funding_total_usd'].quantile(.99)
# lower_lim = copy_data['funding_total_usd'].quantile(.01)
# copy_data = copy_data[(copy_data['funding_total_usd'] < upper_lim) & (copy_data['funding_total_usd'] > lower_lim)]


# 4
# Finding Class value and separating from data
# working on missing value
copy_data['status'].fillna(copy_data['status'].mode().values[0], inplace=True)
# Encoding Class Variable
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(array(copy_data.iloc[:, 1]))
Y = Y.reshape(-1, 1)
Y = pd.DataFrame.from_records(Y)
Y = Y.rename(columns={0: 'Y'})
Y.to_csv(r'C:\IBA\Foundation to Data Science\Project FInal\Y.csv', index=False)
copy_data = copy_data.drop(columns=['status'])


# 5
# Data Pre processing
# Filling Missing values For Market
copy_data['market'].fillna(copy_data['market'].mode().values[0], inplace=True)
# Hot Encoding market
encoded_market = label_encoder.fit_transform(array(copy_data.iloc[:, 0]))
# Adding back Market in Data again
copy_data = copy_data.assign(market=encoded_market)

# Filling Missing values For Country
copy_data['country_code'].fillna(copy_data['country_code'].mode().values[0], inplace=True)
# Hot Encoding Country
encoded_country = label_encoder.fit_transform(array(copy_data.iloc[:, 1]))
# Adding back Country in Data again
copy_data = copy_data.assign(country_code=encoded_country)

# Filling Missing values For Region
copy_data['region'].fillna(copy_data['region'].mode().values[0], inplace=True)
# Hot Encoding Region
encoded_region = label_encoder.fit_transform(array(copy_data.iloc[:, 2]))
# Adding back Region in Data again
copy_data = copy_data.assign(region=encoded_region)

# Data Pre Processing is complete

# Saving Data
X = copy_data
X.to_csv(r'C:\IBA\Foundation to Data Science\Project FInal\X.csv', index=False)

# Heat maps
plt.figure(1, figsize=(25, 12))
cor = sns.heatmap(X.corr(), annot=True)
plt.savefig('HeatMap')
plt.show()
