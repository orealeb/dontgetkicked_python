#!/usr/bin/env python
import csv




def main():
	#input training and test dataset
    with open('/home/ore/Documents/LiClipse Workspace/Kaggle/data/training.csv', 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',') #delimiter is ',' seperating fields
        datareader.next() #skip header and read only data
        training_data = []
        for row in datareader:
            training_data.append(row)
    with open('home/ore/Documents/LiClipse Workspace/Kaggle/data/test.csv', 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',') #delimiter is ',' seperating fields
        datareader.next() #skip header and read only data
        test_data = []
        for row in datareader:
            test_data.append(row)

    #get label 'IsBadBuy' column and remove 'RefId' for both datasets, store test_data_refid for submission
    training_data_no_label = [line[2 : :] for line in training_data]
    training_data_label = [int(line[1]) for line in training_data]
    test_data = [line[1 : :] for line in test_data]
    test_data_refid = [line[0] for line in test_data]

    #extract the year and month
	#insert months from purchase date as a new column
	#change purchase date to only contain the year since days is useless
    idx = 0
    for i in range(len(training_data_no_label)):
    	year = training_data_no_label[i][idx].rfind('/', 0, len(training_data_no_label[i][idx]))
    	month = training_data_no_label[i][idx].find('/', 0, len(training_data_no_label[i][idx]))
    	x[i].append(training_data_no_label[i][idx][0 : month])
    	training_data_no_label[i][idx] = training_data_no_label[i][idx][year + 1 : :]

    idx = 0
    for i in range(len(test_data)):
    	year = test_data[i][idx].rfind('/', 0, len(test_data[i][idx]))
    	month = test_data[i][idx].find('/', 0, len(test_data[i][idx]))
    	test_data[i].append(test_data[i][idx][0 : month])
    	test_data[i][idx] = test_data[i][idx][year + 1 : :]


    #fill empty number values with median of values in its column
    numbers_columns = [3, 12, 16, 17, 18, 19, 20, 21, 22, 23, 29, 31]

    for idx in numbers_columns:
    	for i in range(len(training_data_no_label)):
        if training_data_no_label[i][idx] == '' or training_data_no_label[i][idx] == 'NULL':
        	training_data_no_label[i][idx] = median(training_data_no_label[:,idx])
        else:
            training_data_no_label[i][idx] = float(training_data_no_label[i][idx])

	for idx in numbers_columns:
    	for i in range(len(test_data)):
        if test_data[i][idx] == '' or test_data[i][idx] == 'NULL':
        	test_data[i][idx] = median(test_data[:,idx])
        else:
            test_data[i][idx] = float(test_data[i][idx])


    #convert all categories to integers
    cate_feature_indices = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15,
                            24, 25, 26, 27, 28, 30, 32]
    for column in cate_feature_indices:
    	training_data_no_label = categorize(training_data_no_label, column)
    	test_data = categorize(test_data, column)


    #done with preprocessing





def categorize(x, column):
	categories = {}
	count=0
	for row in x:
		cat = row[column]
		if not car in categories:
			categories[cat] = count
			count+=1
	for  i in range(len(x)):
		x[i][column] = categories[cat]
	return x




def median(column_values):
    sortedLst = sorted(column_values)
    lstLen = len(columnValues)
    index = (lstLen - 1) // 2
    if (lstLen % 2):
        return sortedLst[index]
    else:
        return (sortedLst[index] + sortedLst[index + 1])/2.0




main()