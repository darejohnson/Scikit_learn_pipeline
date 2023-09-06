import joblib
from load_data import load_tips_data
from preprocess_data import PreprocessingTransformer
from train_model import TrainAndEvaluateModelTransformer
from hyperparameter_tuning import GridSearchCV, RandomForestClassifier

def serialize_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model serialized and saved as '{filename}'")

if __name__ == "__main__":
    data = load_tips_data()
    X = PreprocessingTransformer().fit_transform(data)
    y = (data['tip'] > 5).astype(int)
    trained_model = TrainAndEvaluateModelTransformer().fit(X, y)
    tuned_model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid={
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }, cv=5).fit(X, y)
    serialize_model(tuned_model, "best_model.pkl")
