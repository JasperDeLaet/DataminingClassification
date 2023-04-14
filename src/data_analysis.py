import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns

data_df = pd.read_excel('./data/existing-customers.xlsx')

total = data_df['race'].value_counts()
percentage = round((data_df['race'].value_counts()/len(data_df))*100)
race_count = pd.concat([total, percentage], axis=1, keys=['Total', '%'])
print(race_count)

# grid = sns.FacetGrid(data_df, col='class', row='race', aspect=1.6)
# grid.map(plt.hist, 'age', alpha=.5, bins=20)
# grid.add_legend()
# plt.savefig("race_analysis.pdf", format="pdf", bbox_inches="tight")
