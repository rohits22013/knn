import pandas as pd
import numpy as np
content = pd.read_csv("Data_for_UCI_named.csv")
print(content.head())

# print(len(content['stabf'].unique()))

income = [[] for x in range(len(content))]

cont = content.values
#print(cont)
i = 0
for each in content.head(0):
    if each == "stabf":
        temp = content[each].unique()
        d = {temp[x]:x for x in range(len(temp))}
        for j in range(len(cont)):
            income[j].append(d[cont[j][i]])
    else:
        for j in range(len(cont)):
            income[j].append(cont[j][i])
    i += 1
#print(income)
x_train = [[income[i][j] for j in range(0, len(income[i])-1)] for i in range(0, 7500)]
y_train = [income[i][13] for i in range(0, 7500)]
x_test = [[income[i][j] for j in range(0, len(income[i])-1)] for i in range(7500, len(income))]
y_test = [income[i][13] for i in range(7500, len(income))]

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

def euclidenDistance(x1, x2):
    distance = 0
    l = len(x1)
    for i in range(l):
        distance += (x1[i] - x2[i])**2
    return distance**(1/2)

def knn(xtrain, ytrain, xtest, k):
    trainl = len(xtrain)
    testl = len(xtest)
    res = []
    for i in range(testl):
        distance = []
        for j in range(trainl):
            dis = euclidenDistance(xtrain[j], xtest[i])
            #print(dis)
            if len(distance) < k:
                distance.append([dis, ytrain[j]])
            elif max([x[0] for x in distance]) > dis:
                ind = [x[0] for x in distance].index(max(x[0] for x in distance))
                distance[ind] = [dis, ytrain[j]]
        temp = [x[1] for x in distance]
        d = {i:temp.count(i) for i in range(0,2)}
        key = 0
        for i in range(len(d)):
            if d[i] > d[key]:
                key = i
        res.append(key)
        
    return res
y_pred = knn(x_train, y_train, x_test,14)
def accuracyTest(res, old):
    cr = 0
    for i in range(len(res)):
        if res[i] == old[i]:
            cr += 1
    print("Accuracy is: ", float(cr/len(res)*100),"%")
accuracyTest(y_pred, y_test)
