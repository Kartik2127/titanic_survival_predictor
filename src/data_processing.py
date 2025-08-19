import pandas as pd
def load_data(path):
    return pd.read_csv(path)

def impute_embarked(df):
    mode=df['Embarked'].mode()[0]
    df['Embarked']=df['Embarked'].fillna(mode)
    return df


def impute_age(df):
    median=df['Age'].median()
    df['Age']=df['Age'].fillna(median)
    return df 

def preprocess_features(df):
    df=df.drop(columns=["Name","PassengerId","Cabin","Ticket"])
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df=pd.get_dummies(df, columns=['Embarked'], drop_first=True,dtype=int)
    return df



