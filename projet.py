import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import itertools

tipsPercent = []
featureNames = ["vendor_id", "passenger_count", "trip_distance", "rate_code", "payment_type", "extra", "mta_tax", "imp_surcharge", "tolls_amount", "pickup_location_id", "dropoff_location_id", "pickup_month", "pickup_day", "pickup_time", "dropoff_time"]

nbDataToUse = 9000000 #max = 10 000 000
#nbDataToUse = 1000

data = np.zeros((nbDataToUse, len(featureNames)))

dataPath = '../data/taxi_trip_data.csv'

with open(dataPath, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    i = 0
    for row in csv_reader:
        if float(row["fare_amount"]) != 0:
            i += 1
        if (i%100000 == 0):
            print(f"Data loading : {i/nbDataToUse*100}%")
        if (i > nbDataToUse):
            break
        if float(row["fare_amount"]) != 0:
            tipsPercent.append(float(row["tip_amount"]) / float(row["fare_amount"]) * 100)
            singleData = []
            singleData.append(row["vendor_id"])
            singleData.append(row["passenger_count"])
            singleData.append(row["trip_distance"])
            singleData.append(row["rate_code"])
            singleData.append(row["payment_type"])
            singleData.append(row["extra"])
            singleData.append(row["mta_tax"])
            singleData.append(row["imp_surcharge"])
            singleData.append(row["tolls_amount"])
            singleData.append(row["pickup_location_id"])
            singleData.append(row["dropoff_location_id"])

            pickup_datetime = (row["pickup_datetime"]).split("-")
            singleData.append(int(pickup_datetime[1]))
            singleData.append(int(pickup_datetime[2][0:2]))
            pickup_time = pickup_datetime[2][3:].split(":")
            timeInSeconds = int(pickup_time[0])*3600 + int(pickup_time[1])*60 + int(pickup_time[2])
            singleData.append(timeInSeconds)

            dropoff_datetime = (row["dropoff_datetime"]).split("-")
            dropoff_time = dropoff_datetime[2][3:].split(":")
            timeInSeconds = int(dropoff_time[0])*3600 + int(dropoff_time[1])*60 + int(dropoff_time[2])
            singleData.append(timeInSeconds)

            #singleData = np.array(singleData)
            data[i-1] = singleData

tipsPercent = np.array(tipsPercent)
maxTipProcessed = 40
data = np.array(data)
data = minmax_scale(data)

pricePrecisionPercent = 10
nbClasses = int(40/pricePrecisionPercent) + 1

target = np.floor(tipsPercent/pricePrecisionPercent)
target[target >= 40/pricePrecisionPercent] = nbClasses-1

classesText = []
classesCount = []

for i in range(nbClasses):
    classesText.append(str(i*pricePrecisionPercent))
    classesCount.append(np.count_nonzero(target == i))
classesCount = np.array(classesCount)

#plt.bar(classesText, classesCount)
#plt.show()

lowestClassCount = np.min(classesCount) 
#lowestClassCount = 1000
lowestClassCount = 250

balancedData = []
balancedTarget = []
for i in range(nbClasses):
    dataOfClass = data[target==i]
    randomIndexes = np.random.randint(0, len(dataOfClass), lowestClassCount)
    balancedData.extend(data[target==i][randomIndexes])
    balancedTarget.extend(target[target==i][randomIndexes])
balancedData = np.array(balancedData)
balancedTarget = np.array(balancedTarget)

classesText = []
classesCount = []

for i in range(nbClasses):
    classesText.append(str(i*pricePrecisionPercent))
    classesCount.append(np.count_nonzero(balancedTarget == i))
classesCount = np.array(classesCount)

#plt.bar(classesText, classesCount)
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(balancedData, balancedTarget, test_size=0.25)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33)

#C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
#gamma_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]


################################################################################
# This block iterates on every possible combination of features and test an SVM on it. It takes several hours to complete
# So keep that in mind if you uncomment it. The best result obtained was about 0.5 in test
"""
C_list = [0.001, 1, 1000]
gamma_list = [0.001, 1, 1000]
featureNumbers = np.arange(0, len(featureNames))

result_subset_train = []
result_subset_valid = []
result_subset_test = []
for featuresLength in range(1, len(featureNumbers) + 1):
    for subset in itertools.combinations(featureNumbers, featuresLength):
        X_train_subset = X_train[:,subset]
        X_valid_subset = X_valid[:,subset]
        X_test_subset = X_test[:,subset]
        results_train = []
        results_valid = []
        results_test = []
        print(subset)
        for C in C_list:
            for gamma in gamma_list:
                clf_svc = SVC(C=C, kernel="rbf", gamma=gamma)
                clf_svc.fit(X_train_subset, y_train)
                #print("training done")
                train_precision = clf_svc.score(X_train_subset, y_train)
                #print(train_precision)
                valid_precision = clf_svc.score(X_valid_subset, y_valid)
                #print(valid_precision)
                test_precision = clf_svc.score(X_test_subset, y_test)
                #print(test_precision)
                results_train.append(train_precision)
                results_valid.append(valid_precision)
                results_test.append(test_precision)

        results_train = np.array(results_train)
        results_test = np.array(results_test)
        results_valid = np.array(results_valid)
        result_subset_train.append(np.max(results_train))
        result_subset_test.append(np.max(results_test))
        result_subset_valid.append(np.max(results_valid))

print(result_subset_train)
print(result_subset_valid)
print(result_subset_test)

result_subset_train = np.array(result_subset_train)
result_subset_valid = np.array(result_subset_valid)
result_subset_test = np.array(result_subset_test)

print(np.max(result_subset_train))
print(np.max(result_subset_valid))
print(np.max(result_subset_test))
"""
################################################################################

"""
result_subset_train = []
result_subset_valid = []
result_subset_test = []
for i, feature in enumerate(featureNames):
    X_train_subset = X_train[:,i].reshape(-1, 1)
    X_valid_subset = X_valid[:,i].reshape(-1, 1)
    X_test_subset = X_test[:,i].reshape(-1, 1)
    results_train = []
    results_valid = []
    results_test = []
    print(i)
    for C in C_list:
        for gamma in gamma_list:
            clf_svc = SVC(C=C, kernel="rbf", gamma=gamma)
            clf_svc.fit(X_train_subset, y_train)
            #print("training done")
            train_precision = clf_svc.score(X_train_subset, y_train)
            #print(train_precision)
            valid_precision = clf_svc.score(X_valid_subset, y_valid)
            #print(valid_precision)
            test_precision = clf_svc.score(X_test_subset, y_test)
            #print(test_precision)
            results_train.append(train_precision)
            results_valid.append(valid_precision)
            results_test.append(test_precision)

    results_train = np.array(results_train)
    results_test = np.array(results_test)
    results_valid = np.array(results_valid)
    result_subset_train.append(np.max(results_train))
    result_subset_test.append(np.max(results_test))
    result_subset_valid.append(np.max(results_valid))

#print(result_subset_train)
#print(result_subset_valid)
#print(result_subset_test)

for i, feature in enumerate(featureNames):
    print(f"Résultats pour {feature}")
    print(f"Train : {result_subset_train[i]}")
    print(f"Valid : {result_subset_valid[i]}")
    print(f"Test : {result_subset_test[i]}")

"""
"""
y = np.random.random(len(data))
scatter = plt.scatter(data, y, cmap=plt.cm.Paired,\
        c=target)
plt.show()
"""