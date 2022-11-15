import pandas as pd # to read and manipulating data 
import numpy as np # to calculate mean and standard deviations

df = pd.read_csv('Fertilizer_Prediction.csv')
dff = pd.read_csv('Fertilizer_Prediction.csv')

df.columns #['Temperature', 'Humidity', 'Rainfall', 'pH', 'N', 'P', 'K', 'Soil','Crop', 'Fertilizer'],
# print(df.describe(include='object'))

from sklearn.preprocessing import MinMaxScaler # to normalize data
from sklearn.preprocessing import LabelEncoder # to encode object variable to numeric
from sklearn.model_selection import train_test_split # to split data into trainin

#Label Encoding 
le = LabelEncoder()
df['Fertilizer']= le.fit_transform(df['Fertilizer'])
df['Soil']= le.fit_transform(df['Soil'])
df['Crop']= le.fit_transform(df['Crop'])

X = df.drop(['Fertilizer'], axis=1) #feature variables
y = df[['Fertilizer']] #Target variable

# Create train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
from sklearn.tree import DecisionTreeClassifier #to build a classification tree

#Decision Tree model
#random state (int): Controls the randomness of the estimator for reproducibility
model = DecisionTreeClassifier(random_state=42)

# Train the model using the training sets
model = model.fit(X_train, y_train)

# reading the input from the file
lst =[]
with open('input.txt', 'r') as f:
    for i in f:
        data=i
        lst = list(data.split(","))
        
f.close()


# converting the input list to dataframe rwo
frame = pd.DataFrame([lst])

# predicting on the new data
# y_pred = model.predict(frame)

# converting numpy array to string for writing
# import numpy as np
# y_pred = np.array_str(y_pred)




if y_pred == 0:
    y_pred= "DAP"
elif y_pred == 1:
    y_pred = "DAP & MOP"
elif y_pred == 2:
    y_pred = "Good NPK"
elif y_pred == 3:
    y_pred = "MOP"
elif y_pred == 4:
    y_pred = "Urea"
elif y_pred == 5:
    y_pred = "Urea & DAP"
else:
    y_pred = "Urea & MOP"
    
    
# writting the output in file
f_out = open('output.txt', 'w')

f_out.write(y_pred)
f_out.close()



# Predict the response for test dataset
y_pred = model.predict(X_test)




from sklearn.metrics import accuracy_score, classification_report # to calcutate accuracy of model
from sklearn.metrics import plot_confusion_matrix # to draw confusion_matrix
# print(y_pred_DT)
# # Model Accuracy, how often is the classifier correct?
print('Accuracy: ', accuracy_score(y_test, y_pred))
# #Classification report
# print(classification_report(y_test, y_pred_DT))







