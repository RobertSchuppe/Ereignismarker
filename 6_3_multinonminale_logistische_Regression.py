import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

df = pd.read_csv('Fragebogen.csv')
text = 'mountain'

version_col = 'version_' + text
answer_col = 'answer_' + text

df = df.dropna(subset=[version_col, answer_col])

df.replace('below 18', '19-40', inplace=True)
df = df[df['gender'] != 'keine Angabe']
df = df[df['gender'] != 'divers']

df['factor_A_B'] = df[version_col].str[0]
df['factor_1_2'] = df[version_col].str[1]

df[answer_col] = df[answer_col].astype('category')
df[answer_col] = df[answer_col].cat.reorder_categories(['both', 'x', 'y', 'neither'], ordered=False)

print('Answer categories:', df[answer_col].cat.categories.tolist())

X = pd.get_dummies(df[['version_' + text, 'gender', 'age']])
X = X.drop(columns=['version_' + text + '_A1', 'version_' + text + '_A2', 'gender_female', 'age_19-40'])

X = sm.add_constant(X)

y = df[answer_col]

model = sm.MNLogit(y, X)
result = model.fit()

print(result.summary())

marginal_effects = result.get_margeff()

print(marginal_effects.summary())