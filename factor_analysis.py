import pandas
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import numpy

dataset = pandas.read_csv("dataset_final.csv")
q_data = dataset[dataset.columns[~dataset.columns.isin(['math'])]]

## tests difference from identity matrix
chi2 ,p=calculate_bartlett_sphericity(dataset)
print(chi2, p)
## p = 0.0

## test for n=4 to start
# machine = FactorAnalyzer(n_factors=40, rotation=None)
# machine.fit(dataset)
# ev, v = machine.get_eigenvalues()
# print(ev)
## the first 5 factors have variances > 1; these are likely the 5 groups we want to analyze

# machine = FactorAnalyzer(n_factors=6, rotation=None)
# machine.fit(dataset)
# output = machine.loadings_
# print(output)

# machine = FactorAnalyzer(n_factors=5, rotation=None)
# machine.fit(dataset)
# output = machine.loadings_
# print(output)

# machine = FactorAnalyzer(n_factors=4, rotation=None)
# machine.fit(dataset)
# output = machine.loadings_
# print(output)

machine = FactorAnalyzer(n_factors=6, rotation='varimax')
machine.fit(dataset)
factor_loadings = machine.loadings_
print(factor_loadings)

machine = FactorAnalyzer(n_factors=5, rotation='varimax')
machine.fit(dataset)
factor_loadings = machine.loadings_
print(factor_loadings)

machine = FactorAnalyzer(n_factors=4, rotation='varimax')
machine.fit(dataset)
factor_loadings = machine.loadings_
print(factor_loadings)









