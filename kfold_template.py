import pandas
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def run_kfold(machine, data, target, n, continuous):

	kfold_object = KFold(n_splits=n)
	kfold_object.get_n_splits(data)

	all_return_values = []

	i=0
	for train_index, test_index in kfold_object.split(data):
		i = i+1
		print("Round", str(i))
		print("Training index: ")
		print(train_index)
		print("Test index:")
		print(test_index)

		data_train = data[train_index]
		target_train = target[train_index]
		data_test = data[test_index]
		target_test = target[test_index]

		machine.fit(data_train, target_train)
		prediction = machine.predict(data_test)

		if (continuous == True):
			r2 = metrics.r2_score(target_test, prediction)
			print("R2 =:", r2)
			print("\n\n")
			all_return_values.append(r2)
		else:
			accuracy_score = metrics.accuracy_score(target_test, prediction)
			print("accuracy score =:", accuracy_score)
			all_return_values.append(accuracy_score)

			confusion_matrix = metrics.confusion_matrix(target_test, prediction)
			print("confusion matrix: ", confusion_matrix)
			print("\n\n")