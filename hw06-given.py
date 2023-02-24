import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter



# This function mutates, and also returns, the targetDF DataFrame.
# Mutations are based on values in the sourceDF DataFrame.
# You'll need to write more code in this function, to complete it.
def preprocess(targetDF, sourceDF):
    # For the Sex attribute, replace all male values with 0, and female values with 1.
    # (For this historical dataset of Titanic passengers, only "male" and "female" are listed for sex.)
    targetDF.loc[:, "Sex"] = targetDF.loc[:, "Sex"].map(lambda v: 0 if v=="male" else v)
    targetDF.loc[:, "Sex"] = targetDF.loc[:, "Sex"].map(lambda v: 1 if v=="female" else v)
    
    # Fill not-available age values with the median value.
    targetDF.loc[:, 'Age'] = targetDF.loc[:, 'Age'].fillna(sourceDF.loc[:, 'Age'].median())
    
	# -------------------------------------------------------------
	# Problem 4 code goes here, for fixing the error
    targetDF.loc[:, "Embarked"] = targetDF.loc[:, "Embarked"].map(lambda v: 0 if v=="C" else v)
    targetDF.loc[:, "Embarked"] = targetDF.loc[:, "Embarked"].map(lambda v: 1 if v=="Q" else v)
    targetDF.loc[:, "Embarked"] = targetDF.loc[:, "Embarked"].map(lambda v: 2 if v=="S" else v)
    targetDF.loc[:, 'Embarked'] = targetDF.loc[:, 'Embarked'].fillna(sourceDF.loc[:, 'Embarked'].mode().iloc[0])
    
    # -------------------------------------------------------------
	# Problem 5 code goes here, for fixing the error
    # targetDF.loc[:, 'SibSp'] = targetDF.loc[:, 'SibSp'].fillna(sourceDF.loc[:, 'SibSp'].median())
    # targetDF.loc[:, 'Sex'] = targetDF.loc[:, 'Sex'].fillna(sourceDF.loc[:, 'Sex'].median())
    # targetDF.loc[:, 'Age'] = targetDF.loc[:, 'Age'].fillna(sourceDF.loc[:, 'Age'].median())
    # targetDF.loc[:, 'Pclass'] = targetDF.loc[:, 'Pclass'].fillna(sourceDF.loc[:, 'Pclass'].median())
    # targetDF.loc[:, 'Parch'] = targetDF.loc[:, 'Parch'].fillna(sourceDF.loc[:, 'Parch'].median())
    targetDF.loc[:, 'Fare'] = targetDF.loc[:, 'Fare'].fillna(sourceDF.loc[:, 'Fare'].median())
	
# You'll need to write more code in this function, to complete it.
def buildAndTestModel():
    titanicTrain = pd.read_csv("data/train.csv")
    preprocess(titanicTrain, titanicTrain)
    
    predictor = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']
    target = ["Survived"]
    
    inputDF = titanicTrain.loc[:, predictor]
    outputSeries = titanicTrain.loc[:, target]
	
	# -------------------------------------------------------------
	# Problem 4 code goes here, to make the LogisticRegression object.
    lr = LogisticRegression(solver ='liblinear')
    # lr.fit(inputDF)
    # lr.predict()
    
    # scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True) 
    accuracies = model_selection.cross_val_score(lr, inputDF, outputSeries,
                                             cv=3).mean()
    print(accuracies)
    
	
	# -------------------------------------------------------------
	# Problem 5 code goes here, to try the Kaggle testing set
    titanicTest = pd.read_csv("data/test.csv")
    preprocess(titanicTest, titanicTrain)
	
    newlr = LogisticRegression(solver ='liblinear').fit(inputDF, outputSeries)
    predictions = newlr.predict(titanicTest.loc[:,predictor])
    print(predictions, Counter(predictions), sep="\n")

    submitDF = pd.DataFrame(
                {"PassengerId": titanicTest.loc[:,"PassengerId"],
                     "Survived": predictions }
                )
    submitDF.to_csv("data/choi-contaldi-hw06.csv", index=False)
	
	
def test06():
    buildAndTestModel()    
