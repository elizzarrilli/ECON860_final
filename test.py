import pandas
from sklearn import linear_model
import kfold_template

dataset = pandas.read_csv("data/dataset_final.csv")
dataset = dataset.sample(frac=1)

target = dataset.iloc[:,40].values
data = dataset.iloc[:,0:40].values

machine = linear_model.LinearRegression()

return_values = kfold_template.run_kfold(machine, data, target, 4, True)

print(return_values) ## R2 ~ .97 in each round
machine.fit(data,target)

#print(f"coefficients: {machine.coef_}") ## looking at coefficients, cluster 3 has the highest math scores