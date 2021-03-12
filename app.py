from flask import Flask, render_template, request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.show()
train = pd.read_csv('train.csv')

#//////////////////MACHINE LEARNING CODE

def imput_age(cols):
    Age = cols[0]
    Pclass = [1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age'] = train[['Age', 'Pclass']].apply(imput_age, axis = 1)

#Cabin, drop it bacause there are to many gaps
train.drop('Cabin', axis = 1, inplace = True)
#Name and Ticket, drop it bacause are useless information.
train.drop(['Name', 'Ticket'], axis = 1, inplace = True)
#Sex, is necessary to convert categorical features to dummy variables(numbers)
sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)
Train = pd.concat([train, sex, embark], axis = 1)
Train.drop(['Sex', 'Embarked'], axis = 1, inplace = True)

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(Train.drop('Survived', axis = 1), Train['Survived'], test_size = 0.40, random_state = 101)  

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_Train, Y_Train)
predictions = logmodel.predict(X_Test)

#/////////////////////////////////////////

app = Flask(__name__)

@app.route('/')
def show_predict_stock_form():
    return render_template('predictorform.html')

@app.route('/results', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
      #write your function that loads the model
      #model = get_model() #you can use pickle to load the trained model

      PassengerId = request.form['PassengerId']
      Pclass = request.form['Pclass']
      Age = request.form['Age']
      SibSp = request.form['SibSp']
      Parch = request.form['Parch']
      Fare = request.form['Fare']
      male = request.form['male']
      Q = request.form['Q']
      S = request.form['S']

      input_variables = pd.DataFrame([[PassengerId, Pclass, Age, SibSp, Parch, Fare, male, Q, S]], columns=['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S'], dtype=float)

      predicted_survival_rate = logmodel.predict(input_variables)
      return render_template('resultsform.html', predicted_value=predicted_survival_rate)

if __name__ == "__main__":
    app.run(debug = True)