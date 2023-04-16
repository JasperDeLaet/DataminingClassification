
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

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

# Drop education as education and education-num are the same
data_df = data_df.drop(columns=['education'])
data_df = data_df.drop(columns=['native-country'])
data_df = data_df.drop(columns=['race'])
data_df = data_df.drop(columns=['relationship'])

# Drop rows where native country has missing values (because only 1.5 percent of native county values are missing)
# data_df = data_df.dropna(subset=['native-country'])

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
occupations = {"Tech-support": 0,  "Craft-repair": 1, "Other-service": 2, "Sales": 3, "Exec-managerial": 4,
               "Prof-specialty": 5, "Handlers-cleaners": 6, "Machine-op-inspct": 7, "Adm-clerical": 8,
               "Farming-fishing": 9, "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12,
               "Armed-Forces": 13}
data_df['occupation'] = data_df['occupation'].map(occupations)

# Convert workclass into numerical value and solve missing values

workclass_total = data_df['workclass'].value_counts()
workclass_total_dict = workclass_total.to_dict()
workclass_missing = data_df[data_df['workclass'].isna()]
workclass_probabilities = []
for value in list(workclass_total_dict.values()):
    workclass_probabilities.append(value/sum(list(workclass_total_dict.values())))
for index, row in workclass_missing.iterrows():
    random_workclass = np.random.choice(list(workclass_total_dict.keys()), p=workclass_probabilities)
    data_df.loc[data_df['RowID'] == row['RowID'], 'workclass'] = random_workclass

workclasses = {"Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2, "Federal-gov": 3, "Local-gov": 4,
               "State-gov": 5, "Without-pay": 6, "Never-worked": 7}
data_df['workclass'] = data_df['workclass'].map(workclasses)

# Convert marital-status
marital_statuses = {"Married-civ-spouse": 0, "Divorced": 1, "Never-married": 2, "Separated": 3, "Widowed": 4,
                    "Married-spouse-absent": 5, "Married-AF-spouse": 6}
data_df['marital-status'] = data_df['marital-status'].map(marital_statuses)

data_df = data_df[data_df['age'] > 17]

data_df.loc[(data_df['age'] > 17) & (data_df['age'] <= 24), 'age'] = 1
data_df.loc[(data_df['age'] > 24) & (data_df['age'] <= 32), 'age'] = 2
data_df.loc[(data_df['age'] > 32) & (data_df['age'] <= 40), 'age'] = 3
data_df.loc[(data_df['age'] > 40) & (data_df['age'] <= 50), 'age'] = 4
data_df.loc[(data_df['age'] > 50) & (data_df['age'] <= 60), 'age'] = 5
data_df.loc[data_df['age'] > 60, 'age'] = 6

data_df.info()

# I choose to categorize into two categories as the amount of non-zero capital-gain samples is relatively low
# compared to equal to zero
data_df.loc[data_df['capital-gain'] < 1, 'capital-gain'] = 0
data_df.loc[data_df['capital-gain'] > 1, 'capital-gain'] = 1

# Same for capital-loss
data_df.loc[data_df['capital-loss'] < 1, 'capital-loss'] = 0
data_df.loc[data_df['capital-loss'] > 1, 'capital-loss'] = 1

# After analysis and preprocessing we need to split the dataset in test and train sets

shuffled_df = data_df.sample(frac=1, random_state=4)
high_income_df = shuffled_df.loc[shuffled_df['class'] == 1]

low_income_df = shuffled_df.loc[shuffled_df['class'] == 0].sample(n=17000, random_state=42)

normalized_df = pd.concat([high_income_df, low_income_df])

from imblearn.over_sampling import SMOTE
su = SMOTE(random_state=42)


X = normalized_df.drop(columns=['class', 'RowID'])
y = normalized_df[['class']]

X_su, y_su = su.fit_resample(X, y)

print(y_su['class'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_su, y_su, test_size=0.20, random_state=1)



# Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
GaussianNB(priors=None)
nb_pred = nb.predict(X_test)
# nb_accuracy = accuracy_score(y_true=y_test, y_pred=nb_pred)
# print(nb_accuracy)

# Decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
# dt_accuracy = accuracy_score(y_true=y_test, y_pred=dt_pred)
# print(dt_accuracy)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

target_names = ['<=50K', '>50K']
print(classification_report(y_test, nb_pred, target_names=target_names))
print(classification_report(y_test, dt_pred, target_names=target_names))
print(classification_report(y_test, lr_pred, target_names=target_names))

