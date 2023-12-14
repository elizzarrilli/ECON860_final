import pandas
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import numpy

dataset = pandas.read_csv("data/dataset_final.csv")
## remove observations where there is a math score but no personality data
df_clean = dataset.loc[~(dataset==0).all(axis=1)]
print(len(df_clean)) ## length = 19520
## I am also going to try removing obs with ANY zero answers to see if it changes the number of factors
df_no_zeros = dataset.loc[~(dataset==0).any(axis=1)]
print(len(df_no_zeros)) #length = 15195
math = df_no_zeros['math']
math = math.reset_index()
math = math['math']
df_no_zeros = df_no_zeros[df_no_zeros.columns[~df_no_zeros.columns.isin(['math'])]]

### Run factor analysis on ALL obs (DID NOT SEEM CORRECT)

## tests difference from identity matrix
# chi2 ,p=calculate_bartlett_sphericity(dataset)
# print(chi2, p)
## p = 0.0

# machine = FactorAnalyzer(n_factors=40, rotation=None)
# machine.fit(q_data)
# ev, v = machine.get_eigenvalues()
# print(ev)
## the first 4 factors have eigenvalues > 1; these are likely the 4 groups we want to analyze

# machine = FactorAnalyzer(n_factors=4, rotation=None)
# machine.fit(q_data)
# output = machine.loadings_
# print(output)

## Since personality traits may be correlated we will try with varimax 
# machine = FactorAnalyzer(n_factors=4, rotation='varimax')
# machine.fit(q_data)
# factor_loadings4 = machine.loadings_
#print(factor_loadings4)

# q_data = q_data.values
# results = numpy.dot(q_data, factor_loadings4)


#### Run it with df_no_zeros

# tests difference from identity matrix
# chi2 ,p=calculate_bartlett_sphericity(df_clean)
# print(chi2, p)
# # p = 0.0

# machine = FactorAnalyzer(n_factors=40, rotation=None)
# machine.fit(df_no_zeros)
# ev, v = machine.get_eigenvalues()
# print(ev)
## the first 7 factors have eigenvalues > 1; these are likely the 7 groups we want to analyze

# machine = FactorAnalyzer(n_factors=7, rotation=None)
# machine.fit(df_no_zeros)
# output = machine.loadings_
# print(output)

machine = FactorAnalyzer(n_factors=7, rotation='varimax')
machine.fit(df_no_zeros)
factor_loadings_no_zeros = machine.loadings_
df_loadings = pandas.DataFrame(factor_loadings_no_zeros.round(decimals =1))
print(df_loadings)

results_no_zeros = df_no_zeros.values
results_no_zeros = numpy.dot(results_no_zeros, factor_loadings_no_zeros)

#### Save results

# results = pandas.DataFrame(results).round()
# results.rename(columns={0: 'x1', 1: 'x2', 2: 'x3', 3: 'x4'}, inplace=True)
# print(results)
# results.to_csv("data/results4.csv", index=False)

results_no_zeros = pandas.DataFrame(results_no_zeros).round()
results_no_zeros.rename(columns={0: 'x1', 1: 'x2', 2: 'x3', 3: 'x4', 4:'x5', 5:'x6',6:'x7'}, inplace=True)
results_no_zeros['math'] = math
# print(results_no_zeros)
results_no_zeros.to_csv("data/results_no_zeros.csv", index=False)

df_no_zeros = df_no_zeros.reset_index()
all_data = pandas.concat([df_no_zeros, results_no_zeros], axis=1)
all_data.to_csv("data/all_data.csv")
# print(all_data)
# print(len(all_data))





