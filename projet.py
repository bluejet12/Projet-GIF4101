import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

tipsPercent = []
featureNames = ["vendor_id", "passenger_count", "trip_distance", "rate_code", "payment_type", "extra", "mta_tax", "imp_surcharge", "tolls_amount", "pickup_location_id", "dropoff_location_id", "pickup_month", "pickup_day", "pickup_time", "dropoff_time"]
#data = np.array([])

nbDataToUse = 100000 #max = 10 000 000

data = np.zeros((nbDataToUse, len(featureNames)))
print(data.shape)

with open('./data/taxi_trip_data.csv', mode='r') as csv_file:
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
#print(data)
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
#lowestClassCount = 100
balancedData = []
balancedTarget = []
for i in range(nbClasses):
    balancedData.extend(data[target==i][0:lowestClassCount])
    balancedTarget.extend(target[target==i][0:lowestClassCount])
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

C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
gamma_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

result_feature_train = []
result_feature_valid = []
result_feature_test = []
for i, feature in enumerate(featureNames):
    X_train_feature = X_train[:,i].reshape(-1, 1)
    X_valid_feature = X_valid[:,i].reshape(-1, 1)
    X_test_feature = X_test[:,i].reshape(-1, 1)
    results_train = []
    results_valid = []
    results_test = []
    print(i)
    for C in C_list:
        for gamma in gamma_list:
            clf_svc = SVC(C=C, kernel="rbf", gamma=gamma)
            clf_svc.fit(X_train_feature, y_train)
            #print("training done")
            train_precision = clf_svc.score(X_train_feature, y_train)
            #print(train_precision)
            valid_precision = clf_svc.score(X_valid_feature, y_valid)
            #print(valid_precision)
            test_precision = clf_svc.score(X_test_feature, y_test)
            #print(test_precision)
            results_train.append(train_precision)
            results_valid.append(valid_precision)
            results_test.append(test_precision)

    results_train = np.array(results_train)
    results_test = np.array(results_test)
    results_valid = np.array(results_valid)
    result_feature_train.append(np.max(results_train))
    result_feature_test.append(np.max(results_test))
    result_feature_valid.append(np.max(results_valid))

#print(result_feature_train)
#print(result_feature_valid)
#print(result_feature_test)

for i, feature in enumerate(featureNames):
    print(f"Résultats pour {feature}")
    print(f"Train : {result_feature_train[i]}")
    print(f"Valid : {result_feature_valid[i]}")
    print(f"Test : {result_feature_test[i]}")

"""
y = np.random.random(len(data))
scatter = plt.scatter(data, y, cmap=plt.cm.Paired,\
        c=target)
plt.show()
"""