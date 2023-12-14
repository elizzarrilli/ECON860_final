import pandas
from sklearn import linear_model
import kfold_template

dataset = pandas.read_csv("data/clustered_data.csv")
dataset = dataset.sample(frac=1)

## create categorical variables for each group
dataset = pandas.concat([dataset, pandas.get_dummies(dataset['cluster'])], axis=1); dataset
print(dataset)

### Run initial kfold template for linear regression with 4 splits

target = dataset.iloc[:,8].values
data = dataset.iloc[:,0:7].values

machine = linear_model.LinearRegression()

return_values = kfold_template.run_kfold(machine, data, target, 4, True)

print(return_values) ## R2 ~ .97 in each round
machine.fit(data,target)

## Predict math scores for individuals in the data set
dataset['predicted_score'] = machine.predict(data)
print(dataset['math'], dataset['predicted_score'])

coefficients = {
	'Trait': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'],
	'coefficient': machine.coef_
}
print(pandas.DataFrame(coefficients))


### Run LinReg on the 4 clusters to see which has the greatest impact on math score

# target = dataset.iloc[:,8].values
# data = dataset.iloc[:,9:12].values

# machine = linear_model.LinearRegression()

# return_values = kfold_template.run_kfold(machine, data, target, 4, True)

# print(return_values) ## R2 ~ .97 in each round
# machine.fit(data,target)

# print(f"coefficients: {machine.coef_}") ## looking at coefficients, cluster 3 has the highest math scores













