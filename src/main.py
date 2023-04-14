
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


# Further feature analysis
# low_income = 'low_income'
# high_income = 'high_income'
#
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# women = data_df[data_df['sex'] == 'Female']
# men = data_df[data_df['sex'] == 'Male']
# ax = sns.distplot(women[women['class'] == 0].age.dropna(), label=low_income, ax=axes[0], kde=False)
# ax = sns.distplot(women[women['class'] == 1].age.dropna(), label=high_income, ax=axes[0], kde=False)
# ax.legend()
# ax.set_title('Female')
# ax = sns.distplot(men[men['class'] == 0].age.dropna(), label=low_income, ax=axes[1], kde=False)
# ax = sns.distplot(men[men['class'] == 1].age.dropna(), label=high_income, ax=axes[1], kde=False)
# ax.legend()
# _ = ax.set_title('Male')
# plt.show()
