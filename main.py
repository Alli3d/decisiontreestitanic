import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# get only relevant data
df = pd.read_csv("titanic.csv")
reldf = df.drop(['PassengerId','Survived','Name','SibSp',
                   'Parch','Ticket','Cabin','Embarked'],axis='columns')

# fill NaN with 0
cleandf = reldf.fillna(0)
target = df['Survived']

# label encode sex catergory into 0/1
le_Sex = LabelEncoder()
cleandf['Sex'] = le_Sex.fit_transform(cleandf['Sex'])

# training
model = tree.DecisionTreeClassifier()
model.fit(cleandf, target)
# class, female/male, age, fare
inputclass = input("Class (1st, 2nd, 3rd): ")[0]
inputgender = input("Gender (F/M): ")
inputgenderadj = 0 if inputgender == "F" else 1
inputage = input("Age: ")
inputfare = input("Fare (in £): ")

if model.predict([[inputclass,inputgenderadj,inputage,inputfare]]) == [0]:
    print("\n\nPassenger doesn't survive.\n\n")
else:
    print("\n\nPassenger survives.\n\n")