import pickle
import pandas as pd


data_df = pd.read_excel('./data/potential-customers.xlsx')

data_df = data_df.dropna()

data_df = data_df.drop(columns=['education'])
data_df = data_df.drop(columns=['native-country'])
data_df = data_df.drop(columns=['race'])
data_df = data_df.drop(columns=['relationship'])
genders = {"Male": 0, "Female": 1}
data_df['sex'] = data_df['sex'].map(genders)


occupations = {"Tech-support": 0,  "Craft-repair": 1, "Other-service": 2, "Sales": 3, "Exec-managerial": 4,
               "Prof-specialty": 5, "Handlers-cleaners": 6, "Machine-op-inspct": 7, "Adm-clerical": 8,
               "Farming-fishing": 9, "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12,
               "Armed-Forces": 13}
data_df['occupation'] = data_df['occupation'].map(occupations)

workclasses = {"Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2, "Federal-gov": 3, "Local-gov": 4,
               "State-gov": 5, "Without-pay": 6, "Never-worked": 7}
data_df['workclass'] = data_df['workclass'].map(workclasses)

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

data_df.loc[data_df['capital-gain'] < 1, 'capital-gain'] = 0
data_df.loc[data_df['capital-gain'] > 1, 'capital-gain'] = 1

data_df.loc[data_df['capital-loss'] < 1, 'capital-loss'] = 0
data_df.loc[data_df['capital-loss'] > 1, 'capital-loss'] = 1

X = data_df.drop(columns=['RowID'])

# Load model created in main.py (Random forest model threshold =0.2)
filename = 'random_forest.pickle'
rf = pickle.load(open(filename, "rb"))

# Create predictions using the random forest model trained on the existing-customers data
rf_pred_proba = rf.predict_proba(X)
rf_pred = (rf_pred_proba[:, 1] >= 0.2).astype('bool')

# select the right data (RowIDs)
pred_tested_true = data_df[rf_pred]

row_ids = pred_tested_true['RowID']

# Write all RowIDS to text file
with open('predicted_row_ids.txt', 'w') as f:
    for row in row_ids.items():
        f.write(str(row[1]) + '\n')

