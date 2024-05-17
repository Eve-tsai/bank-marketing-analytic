#bank marketing dataset
#Find the best strategies to improve for the next marketing campaign
#預測客戶是否會訂閱定期存款
#data resource: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data


import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

url = 'https://raw.githubusercontent.com/Eve-tsai/bank-marketing-analytic/main/bank.csv'
df = pd.read_csv(url)
df.info()
df.describe()

#check null
null_counts = df.isna().sum()
print(null_counts)


df.replace("Unknown", np.nan, inplace=True)
df=df.fillna(df.mode().iloc[0])

#data exploration


#deposit rate
plt.figure(figsize=(6,9))   
df.deposit.value_counts()
plt.pie(df['deposit'].value_counts(),labels=df['deposit'].value_counts().index,autopct = "%1.2f%%")
plt.title("deposit rate", {"fontsize" : 20}) 
plt.legend(loc = "best")

#age distribution
plt.figure(figsize=(10,6)) 
age= df["age"].values.tolist()
plt.hist(age, bins=20, color='skyblue', edgecolor='black') 
plt.xlabel('Age')  
plt.ylabel('count')  
plt.title('Age Distribution')  
plt.show()

#job distribution
count=df.job.value_counts()
print(df['job'])
plt.figure(figsize=(10,6)) 
count.plot(kind='bar', legend=False)
plt.xlabel('job')  
plt.ylabel('count')  
plt.title('Job Distribution With Average Age')  
plt.xticks(rotation=45)
plt.show()

#Average Age by Job
plt.xticks(rotation=45)
plt.figure(figsize=(10,6)) 
avg_age= df.groupby('job')['age'].mean().reset_index()
plt.bar(avg_age['job'], avg_age['age'], color='skyblue')
plt.xlabel('Job')
plt.ylabel('Average Age')
plt.title('Average Age by Job')
plt.xticks(rotation=45)
plt.show()



#pre-processing
#variable classification
cate_col = df.select_dtypes(include=['object']).columns
num_col = df.select_dtypes(exclude=['object']).columns



label_encoder = LabelEncoder()
for column in cate_col:
    df[column] = label_encoder.fit_transform(df[column])

scaler = StandardScaler()
df[num_col] = scaler.fit_transform(df[num_col])


'''
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["default"]=le.fit_transform(df["default"])
df["housing"]=le.fit_transform(df["housing"])
df["loan"]=le.fit_transform(df["loan"])
df["deposit"]=le.fit_transform(df["deposit"]) 

from sklearn.preprocessing import scale
df['age']=scale(df['age'])
df['balance']=scale(df['balance'])
df['duration']=scale(df['duration'])
df['campaign']=scale(df['campaign'])
df['pdays']=scale(df['pdays'])
df['previous']=scale(df['previous'])
'''

#variable correlation coefficient

matrix = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(matrix, annot=True, cmap='OrRd', fmt='.2f',linewidths=0.6)
plt.title('Correlation Matrix')
plt.show()


#save model result
result = pd.DataFrame(columns=['Model Name','Accuracy',"Precision"])

#data spliting
X = df.drop(columns=['deposit'])  
y = df['deposit']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=889)

#-----------------------------------------------------------------------------#
#logistic regression#
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_lr = LogisticRegression()
model_lr .fit(X_train_scaled, y_train)

y_pred = model_lr.predict(X_test_scaled)

# Accuracy: 0.7711598746081505
print("Accuracy_lr:", accuracy_score(y_test, y_pred))
# precision: 0.7597712106768351
print("precision_lr:", precision_score(y_test, y_pred))

#save lr result
log_re={'Model Name':'logistic regression','Accuracy':accuracy_score(y_test,y_pred),
       'Precision':precision_score(y_test,y_pred)}
result.loc[len(result)] =log_re

#lr confusion matrix
cm_lr=confusion_matrix(y_test, y_pred)
cm_display_lr=ConfusionMatrixDisplay(confusion_matrix =cm_lr,
                                  display_labels = ['Not Deposite','Deposite'])
cm_display_lr.plot()
plt.title("Confusion Matrix of Logistic Regression Classifier")
plt.show()

#logistic regression roc
y_prob_lr = model_lr.predict_proba(X_test_scaled)[:, 1]
fpr_lr, tpr_lr, thresholds = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure()
plt.plot(fpr_lr,tpr_lr,color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_lr)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()




#---------------------------------------------------------------------------------#
#random forest#
model_rf = RandomForestClassifier(random_state=889)
model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)

# rf accuracy precision
print("Accuracy_rf:", accuracy_score(y_test, y_pred))
print("Precision_rf:",precision_score(y_test, y_pred))

#save rf result
rf_re={'Model Name':'Random Foreset','Accuracy':accuracy_score(y_test,y_pred),
       'Precision':precision_score(y_test,y_pred)}
result.loc[len(result)] =rf_re

#Accuracy: 0.838781907747425
#Precision: 0.7929292929292929


# rf confusion matrix
cm_rf = confusion_matrix(y_test, y_pred)
cm_display_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Not Deposit', 'Deposit'])
cm_display_rf.plot()
plt.title("Confusion Matrix of Random Forest Classifier")
plt.show()

# rf roc
y_prob_rf = model_rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()



#---------------------------------------------------------------------------------#
#SVM#
model_svm = SVC(kernel='linear', C=1.0)
model_svm.fit(X_train, y_train)
y_pred = model_svm.predict(X_test)

#SVM accuracy precison
print("Accuracy_svm:", accuracy_score(y_test, y_pred))
print("Precision_svm:",precision_score(y_test, y_pred))

#save SVM result
svm_re={'Model Name':'SVM','Accuracy':accuracy_score(y_test,y_pred),
       'Precision':precision_score(y_test,y_pred)}
result.loc[len(result)] =svm_re

# SVM confusion matrix
cm_svm = confusion_matrix(y_test, y_pred)
cm_display_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Not Deposit', 'Deposit'])
cm_display_svm.plot()
plt.title("Confusion Matrix of SVM")
plt.show()

# SVM roc
y_prob_svm =model_svm.decision_function(X_test)
fpr_svm, tpr_svm, thresholds = roc_curve(y_test, y_prob_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure()
plt.plot(fpr_svm, tpr_svm, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


#combined roc curve
plt.figure()
plt.plot(fpr_lr,tpr_lr,label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot(fpr_rf,tpr_rf,label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_svm,tpr_svm,label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()










#解讀

'''
ROC曲線（Receiver Operating Characteristic Curve）是用來評估二分類模型性能的工具。以下是ROC曲線的解讀方式：

1. 橫軸 (X軸)：表示假陽性率（False Positive Rate，FPR），計算公式為FPR = FP / (FP + TN)，即錯誤地將負類別預測為正類別的比率。
   
2. 縱軸 (Y軸)：表示真陽性率（True Positive Rate，TPR），也稱為召回率（Recall）或靈敏度（Sensitivity），計算公式為TPR = TP / (TP + FN)，即正確地將正類別預測為正類別的比率。

3. 曲線下方的面積 (AUC)：ROC曲線下的面積（Area Under the Curve, AUC）是評估模型性能的一個數值指標，AUC值越接近1，模型性能越好。AUC = 0.5表示模型的預測能力與隨機猜測相當。

4. 解讀：
   - 靠近左上角：模型性能較好，能夠較好地區分正負類別。
   - 對角線附近：模型性能較差，與隨機猜測無異。

在使用ROC曲線時，可以比較不同模型的AUC值，選擇AUC值更高的模型作為最優模型。
'''

















'''
one: job education marital contact day month poutcome 
    
label(有順序關係): default housing loan deposit
    
scale: age blance duration campaign pdays previous
    
'''





'''
#linear regression
from sklearn.linear_model import LinearRegression

linear_reg=LinearRegression()
linear_reg.fit(X_train,y_train)

y_pred = linear_reg.predict(X_test)
y_test = y_test.to_numpy()

#訊練模型的準確度
from sklearn.metrics import accuracy_score
print('Accuracy: ',accuracy_score(y_test,y_pred))


from sklearn.metrics import accuracy_score
print("測試資料集正確率=",accuracy_score(y_test,y_pred))
'''




'''
#k means
from sklearn.cluster import KMeans
distortion=[]
for i  in range(1,21):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=889,
                  n_init=10,max_iter=200)
    
    kmeans.fit(df)
    distortion.append(kmeans.inertia_) #SSE
print(distortion)    

#find cluster 
import matplotlib.pyplot as plt
plt.plot(range(1,21),distortion,marker="o")
plt.xlabel("Number of clusters")    
plt.ylabel("Distortion") 

#every cluster's center
kmeans2=KMeans(n_clusters=5, init="k-means++", random_state=889,
               n_init=15, max_iter=200)    
    
kmeans2.fit(df) #fit資料集後找值心
centroid=pd.DataFrame(kmeans2.cluster_centers_,columns=df.columns)    
#centroid 1.13898代表高於平均1.13898倍
#類別0,1 以0.5做界線 超過0.5是男 

#多數決預測
X_pred=kmeans2.predict(df)
cross=pd.crosstab(df["deposit"],X_pred)
print(cross)

#k-means accuracy
sum=[]
for i in range(5):
    if cross[i][0]>cross[i][1]:
        sum.append(cross[i][0])
    else:
        sum.append(cross[i][1])
        
count=0
for i in range(len(sum)):
    count +=sum[i]

print(count/len(df))
print("SSE=",kmeans2.inertia_)


k means result
n=8
col_0      0     1    2     3    4    5    6    7
deposit                                          
0        545  1457  643  1119  310  441  786  572
1        695   749  772   967  384  424  637  661
0.565758824583408
SSE= 132161.27055351107


'''







