#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import classification_report



## Load the sample data sets
dataset = pd.read_csv("C:/Users/Istiak/Desktop/Research/testdata.csv")

## Where X is the independent data sets and y is the dependent data sets.
## y depends on the X. y is also called the class label.

X = dataset.drop('class_label', axis=1)
y = dataset['class_label']



## Decision Tree Classifier

from sklearn.model_selection import train_test_split

## Spilt the sample data sets into training and testing data sets. 
## Sample data sets size (SD) = 100%. Training data sets size (TD) = 80%. Testing data sets size (TSD) = 20%.
## TSD = 20%. TD= SD - TSD.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier

## Fits the training and testing datasets in a instance of decision tree classifier class.

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


#Predicts something based on testing data sets.

y_pred_class = classifier.predict(X_test)


#print('Result DTC:',y_pred_class)


from sklearn.metrics import accuracy_score

## Calculate average accuracy

accuracy_class = accuracy_score(y_test, y_pred_class, normalize=True)


print('DTC Accuracy: ', accuracy_class)

## Printing average accuracy

for pred in y_pred_class:
    att_class = accuracy_score(y_test, y_pred_class) 
  # print(pred, att_class )

from sklearn.metrics import classification_report, confusion_matrix


#print(classification_report(y_test, y_pred_class))



## Decision tree Regression

from sklearn.model_selection import train_test_split

## Spilt the sample data sets into training and testing data sets. 
## Sample data sets size (SD) = 100%. Training data sets size (TD) = 80%. Testing data sets size (TSD) = 20%.
## TSD = 20%. TD= SD - TSD.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.tree import DecisionTreeRegressor

## Fits the training and testing datasets in a instance of decision tree regressior class.

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

#Predicts something based on testing data sets.

y_pred_reg = regressor.predict(X_test)



#print('DTR:',y_pred_reg)

from sklearn.metrics import accuracy_score

## Printing average accuracy

accuracy_reg = accuracy_score(y_test, y_pred_reg, normalize=True)
print('DTR Accuracy: ', accuracy_reg)
for result in y_pred_reg:
    res = int(result)
    att1 = (res, accuracy_reg)
   #print(att1)

## If the accuracy of Decision tree classifier (DTC) is greater than Decision tree regressor (DTR), then DTC attack DTR. 
## On the other hand, DTR attacks DTC
## If the Value of DTC = the value of DTR, Then they will not attack each other.


## comparing the value of two result arrays

comp = (y_pred_class==y_pred_reg)

ln = y_pred_reg.size
 
for cmp in comp:
    if cmp == True:
        print('')
    else:
        if (accuracy_class >= accuracy_reg):
            for i in range(0,ln,1):
                
                if y_pred_class[i]!= int(y_pred_reg[i]):
                    print(int(y_pred_class[i]),'Attacks-->',int(y_pred_reg[i]))
                    if accuracy_class  > 0.3:
                        print(int(y_pred_class[i]),'Accepted')
                    elif accuracy_class == 0.3:
                        print(int(y_pred_class[i]),'Undecided')
                    elif accuracy_class < 0.3:
                        print(int(y_pred_class[i]),'Unaccepted')
            
            break
    
     
        else:
            for i in range(0,ln,1):
                
                if y_pred_class[i]!= int(y_pred_reg[i]):
                    print(int(y_pred_reg[i]),'Attacks-->',int(y_pred_class[i]))
                    if accuracy_class > 0.3:
                        print(int(y_pred_reg[i]),'Accepted')
                    elif accuracy_class == 0.3:
                        print(int(y_pred_reg[i]),'Undecided')
                    elif accuracy_class < 0.3:
                        print(int(y_pred_reg[i]),'Unaccepted')
             
            break
            
for cmp in comp:
    if cmp == True:
        print('No Attack')            
            
              
            
#for i in range(0,ln):
    #if (y_pred_class[i] != y_pred_class[i]):
    
        #print(y_pred_class[i],'Attacks -->',int(y_pred_reg[i]) )
        
    
print('Result DTC:',y_pred_class) 
print('DTR:',y_pred_reg) 

print(comp)  

x1=y_pred_class
y1=y_pred_reg

plt.figure(1)
plt.plot(x1,"b-*",label='DTC')
plt.legend(loc='best')
plt.figure(2)
plt.plot(y1,"r-h", label='DTR')
plt.legend(loc='best')
plt.figure(3)
plt.plot(x1,"b-*", label='DTC')
plt.plot(y1,"r-h", label='DTR')
plt.legend(loc=4)
plt.show()

arg = [1,3,3,3,3,3,3,1,2,1,2,1]
plt.plot(arg,"g-*",label='PAF')
plt.legend(loc='best')
 






    


# In[ ]:





# In[ ]:




