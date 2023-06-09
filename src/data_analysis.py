import pandas as pd

# This file is purely meant for data analysis

data_df = pd.read_excel('./data/existing-customers.xlsx')

# Search for missing values
total = data_df.isnull().sum().sort_values(ascending=False)
percent_1 = data_df.isnull().sum()/data_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print(missing_data.head(16))

# Race analysis to see whether race is a good feature to consider for predicting income (looking for bias)
total = data_df['race'].value_counts()
percentage = round((data_df['race'].value_counts()/len(data_df))*100)
race_count = pd.concat([total, percentage], axis=1, keys=['Total', '%'])
# print(race_count)

# sex
total = data_df['sex'].value_counts()
percentage = round((data_df['sex'].value_counts()/len(data_df))*100)
sex_count = pd.concat([total, percentage], axis=1, keys=['Total', '%'])
# print(sex_count)

# workclass
total = data_df['workclass'].value_counts()
percentage = round((data_df['workclass'].value_counts()/len(data_df))*100)
workclass_count = pd.concat([total, percentage], axis=1, keys=['Total', '%'])
# print(workclass_count)

# native country
total = data_df['native-country'].value_counts()
percentage = round((data_df['native-country'].value_counts()/len(data_df))*100)
country_count = pd.concat([total, percentage], axis=1, keys=['Total', '%'])
# print(country_count)

# class
total = data_df['class'].value_counts()
percentage = round((data_df['class'].value_counts()/len(data_df))*100)
class_count = pd.concat([total, percentage], axis=1, keys=['Total', '%'])
print(class_count)

# occupation
total = data_df['occupation'].value_counts()
percentage = round((data_df['occupation'].value_counts()/len(data_df))*100)
occupation_count = pd.concat([total, percentage], axis=1, keys=['Total', '%'])
# print(occupation_count)

# marital-status
total = data_df['marital-status'].value_counts()
percentage = round((data_df['marital-status'].value_counts()/len(data_df))*100)
marital_status_count = pd.concat([total, percentage], axis=1, keys=['Total', '%'])
# print(marital_status_count)
