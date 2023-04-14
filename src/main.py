
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns

data_df = pd.read_excel('./data/existing-customers.xlsx')

# Data analysis
data_df.info()

print(data_df.describe())


total = data_df.isnull().sum().sort_values(ascending=False)
percent_1 = data_df.isnull().sum()/data_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print(missing_data.head(16))

print(data_df.columns.values)

# Converting to numerical values

# Convert income (column 'class') into numerical value
incomes = {"<=50K": 0, ">50K": 1}
data_df['class'] = data_df['class'].map(incomes)

# Convert sex into numerical value
genders = {"Male": 0, "Female": 1}
data_df['sex'] = data_df['sex'].map(genders)

# Solve missing values occupation by assigning them based on the distribution of the not null data
occupation_total = data_df['occupation'].value_counts()
occupation_total_dict = occupation_total.to_dict()

occupation_missing = data_df[data_df['occupation'].isna()]
occupation_probabilities = []
for value in list(occupation_total_dict.values()):
    occupation_probabilities.append(value/sum(list(occupation_total_dict.values())))
for index, row in occupation_missing.iterrows():
    random_occupation = np.random.choice(list(occupation_total_dict.keys()), p=occupation_probabilities)
    data_df.loc[data_df['RowID'] == row['RowID'], 'occupation'] = random_occupation



# Convert occupation into numerical value
# occupations = {"Tech-support": 0,  " Craft-repair": 1, "Other-service": 2, "Sales": 3, "Exec-managerial": 4,
#                "Prof-specialty": 5, "Handlers-cleaners": 6, "Machine-op-inspct": 7, "Adm-clerical": 8,
#                "Farming-fishing": 9, "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12, "Armed-Forces": 13}
# data_df['occupation'] = data_df['occupation'].map(occupations)
