# ECON860_final

This is a walkthrough of how to run the programs to answer the ECON8600 final exam question. 

**Part 1: Factor Analysis**
1. run factor_analysis.py. This is a very long program that I ran in parts so I would comment out parts as I ran it.
2. The code imports the final exam dataset and removes rows with unanswered personality questions. (It also creates a DataFrame with only complete zero rows removed) though I did not ultimately use this in the code.
3. The code tests whether the data is statistically different from an identity matrix and returns the eigenvalues from using n=40.
4. Use the returned eigenvalues to determine the number of factors you should use and generate factors using both no rotation and varimax.
5. It computes the dot product of the factor loadings and the data and saves this in a csv file called 'results_no_zeros.csv' in the data folder.

**Part 2: Clustering**
1. Run cluster.py. This imports the dataset generated above and has code to cluster individuals into 2,3,...,10 clusters using KMeans, KMedoids and Gaussian. It generates silhouette scores of each grouping and plots them. It can be seen that each method peaks at n=4 groups.
2. It then reruns the KMeans clustering code for 4 groups and saves the cluster IDs into the DataFrame for each individual. This dataset is called 'clustered_data.csv' in the data folder.

**Part 3: Supervised Learning Model**
1. For part 3 of the exam, run lin_reg.py to run a linear regression supervised learning model.
2. This runs a linear regression prediction model with 4 splits and generates predicted math scores for all individuals in the dataset.
3. This uses a kfold_template saved as kfold_template.py
4. It also runs a linear regression model of personality traits on math scores and prints the coefficient matrix.

5. * summary_stats.py was a program I used just to describe my data a bit. It was not really used for any specific component of this project.
