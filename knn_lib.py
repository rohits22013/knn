from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
content =pd.read_csv("Data_for_UCI_named.csv")
label_encoder = preprocessing.LabelEncoder() 
print(content.head())

#for each in content.head(0):
content['stabf']= label_encoder.fit_transform(content['stabf'])
content.head()

df=content
print(df.head())

scaler = StandardScaler()

df['p1'].max()
#df['p1'].min()

scaler.fit(df.drop('stabf',axis=1))
scaled_features = scaler.transform(df.drop('stabf',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
print(df_feat.head())

#X=content.iloc[:,0:-1]
#Y=content.iloc[:,-1]


X=df.iloc[:,0:-1]
Y=df.iloc[:,-1]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.25)

classifier=KNeighborsClassifier(n_neighbors=13)
classifier.fit(X_train,Y_train)
pred=classifier.predict(X_test)
accuracy2=accuracy_score(Y_test,pred)
print(accuracy2*100,end='%\n')
print(len(Y_test))
mat=confusion_matrix(Y_test,pred)
print(mat)


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != Y_test))


    plt.figure(figsize=(10,6))
    
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
