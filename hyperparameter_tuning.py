from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from load_data import load_tips_data
from preprocess_data import PreprocessingTransformer  # Import the PreprocessingTransformer
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    # Load and preprocess data
    tips_data = load_tips_data()
    X = tips_data.drop('tip', axis=1)
    y = (tips_data['tip'] > 5).astype(int)

    # Create a pipeline for preprocessing and model training
    model_pipeline = Pipeline([
        ('preprocessor', PreprocessingTransformer()),  # Use the custom preprocessing transformer
        ('model', RandomForestClassifier(random_state=42))
    ])

    # Define hyperparameter grid
    param_grid = {
        'model__n_estimators': [50, 100, 150],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5)
    grid_search.fit(X, y)

    # Get the best tuned model
    best_tuned_model = grid_search.best_estimator_

    print("Best Hyperparameters:", best_tuned_model.named_steps['model'].get_params())
