

import pandas as pd
from pipeline import run_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 

def train_and_evaluate_models():
    processed_df = run_pipeline()
    print("\n--- Starting Model Training & Evaluation ---")

    X = processed_df.drop('Survived', axis=1)
    y = processed_df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Neighbors Classifier": KNeighborsClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(random_state=42),
        "Support Vector Classifier": SVC()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
    
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n--- {name} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, predictions))
        print("-"*(len(name) + 6))

if __name__ == "__main__":
    train_and_evaluate_models()