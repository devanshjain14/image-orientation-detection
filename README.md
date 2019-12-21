## Decisin Tree For Image Orientation

While building a decision tree, we calculate the mean of all the rows for each column. We calculate the expectation which is 4.0 for the current dataset. Since there are only 4 classes and all or of equal frequency so the probability of occurrence of each of them is equal to one quarter. So we get the resultant sum of - p * log2(p) sum for all the 4 classes is 2. For each column, we iterate through all the rows and classify each value as “>=” for being greater than the mean value and “<” for being less than the mean. Then we get entropy for “>=” for all the 4 classes( 0,90,180,270 ) using the count of each class divided by the total number of counts for all the 4 classes as its probability and then applying summation of - p * log2(p) for all the 4 classes. Similarly, we calculate the entropy for the “<” class. After calculating the entropy for the “>=” and “<” we use expectation - Entropy for “>=” and the “<” separately to calculate the information gain for each of the classes. Then we sum up the information gain for both the classes to get the resultant information gain. The column with maximum information gain is chosen. At each step, we append a tuple to ( depth, column index, the mean value of the column ). After that, we split the training array into 2 parts based on the fact whether its column value at that particular index is “>=” or “<” and split its respective labels( angles ) respectively. So, on training these two sets of training arrays and labels separately calculate recursion. After which the two arrays are merged back in order to get the old resultant array if needed.

## Stopping Condition
Whenever the training model increased the depth of 3, we stored the label of the majority class in that split and use that as a classification criterion. So after completing training, we got 2 arrays one with a set of tuples of ( depth, index, mean value ) and the other the array of possible labels in order of split.

## Testing
While testing through the rows, if the depth of the model is equal to maximum depth then we compare it to the mean value if its “>=” then we equate it to the label of the current index else ( index + 1 ) value of the label. And we exit the loop for that row. Else we check. If the current row’s index value is “>=” we continue iterating. Else we check the second value whose depth is equal to current depth + 1 since the first value which comes at the current depth will be for “>=” but we want “<” which will be a second value which comes in the array. During this process, we skip 2 ^ ( max depth - current depth ) in the labels( since it has all the values in the respective order ) and thus we add that to the index value. Then we continue iterating through the loop for the same.

After generating labels for all the rows, we compare the number of matching labels of the predicted labels with the true labels and we get the testing accuracy.

The maximum accuracy achieved using Decision Tree on the current testing data using the training data is 60%.

# How to run the code?

Training : Run the orient.py file from the console with following command line parameters, 

# orient.py train train-data.txt model.txt tree

Testing : Run the orient.py file from the console with following command line parameters, 

# orient.py test test-data.txt model.txt tree




