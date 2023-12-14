import pandas

dataset = pandas.read_csv("data/clustered_data.csv")
dataset = pandas.concat([dataset, pandas.get_dummies(dataset['cluster'])], axis=1); dataset
print(dataset)

sum_stats = dataset[['math', 'cluster']].describe()
print(sum_stats)




