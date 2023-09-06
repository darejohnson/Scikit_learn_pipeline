from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from load_data import load_tips_data
from preprocess_data import PreprocessingTransformer
from sklearn.pipeline import Pipeline

class TrainAndEvaluateModelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        return self

    def transform(self, X):
        y_pred = self.model.predict(X)
        return y_pred

if __name__ == "__main__":
    # Load and preprocess data
    tips_data = load_tips_data()
    preprocessor = PreprocessingTransformer()  # Create an instance without passing data
    preprocessed_data = preprocessor.fit_transform(tips_data)  # Fit and transform
    X = preprocessed_data
    y = (tips_data['tip'] > 5).astype(int)

    # Create a pipeline with TrainAndEvaluateModelTransformer
    model_pipeline = Pipeline([
        ('model', TrainAndEvaluateModelTransformer())
    ])

    # Fit and evaluate the model
    y_pred = model_pipeline.fit_transform(X, y)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {accuracy}")
