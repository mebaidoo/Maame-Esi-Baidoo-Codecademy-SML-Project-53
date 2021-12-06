import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#Loading the data into a dataframe
flags = pd.read_csv("flags.csv", header = 0)

#Investigating the data
print(flags.columns)
print(flags.head())

#Creating a decision tree to classify what Landmass a country is on
#Getting the labels (Landmass) to be used
labels = flags[["Landmass"]]
#Getting the columns containing the colors of flags; using only the colors of a flag to predict or classify the Landmass of a country
data = flags[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange", "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]

#Splitting the dataframes into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

#Making and testing the model
#Calculating the accuracy of the tree 20 times as the size of the tree changes to see what size gives the best prediction accuracy, and plotting in a graph
scores = []
for i in range(1, 21):
  tree = DecisionTreeClassifier(random_state = 1, max_depth = i, max_leaf_nodes = 35)
  tree.fit(train_data, train_labels)
  scores.append(tree.score(test_data, test_labels))

#Plotting the scores
plt.plot(range(1, 21), scores)
plt.show()

#Adding more features (that has to do with shapes in flags) to data, used in classifying Landmass to see if accuracy will increase.
#The depth of the tree isn’t really having an impact on its performance. This might be a good indication that we’re not using enough features.

#Also, setting the max_leaf_nodes to see how accuracy changes (overfitting or underfitting?)