import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# get only relevant data
df = pd.read_csv("titanic.csv")
reldf = df.drop(['PassengerId','Survived','Name','SibSp',
                   'Parch','Ticket','Cabin','Embarked'],axis='columns')

# fill NaN with 0
cleandf = reldf.fillna(0)
print(cleandf.head(10))
target = df['Survived']

# label encode sex catergory into 0/1
le_Sex = LabelEncoder()
cleandf['Sex'] = le_Sex.fit_transform(cleandf['Sex'])
print(cleandf.head(10))

# training
model = tree.DecisionTreeClassifier()
model.fit(cleandf, target)
# class, female/male, age, fare
inputclass = input("Class: ")
inputgender = input("Gender: ")
inputage = input("Age: ")
inputfare = input("Fare: ")
if model.predict([[inputclass,inputgender,inputage,inputfare]]) == [0]:
    print("\n\nPassenger doesn't survive.\n\n")
else:
    print("\n\nPassenger survives.\n\n")