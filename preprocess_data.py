from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from load_data import load_tips_data

class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_features = ['total_bill', 'size']
        self.categorical_features = ['sex', 'smoker', 'day', 'time']

        self.numeric_preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        self.categorical_preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(fill_value='missing', strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', self.numeric_preprocessor, self.numeric_features),
                ('categorical', self.categorical_preprocessor, self.categorical_features)
            ]
        )

    def fit(self, X, y=None):
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

# Create an instance of PreprocessingTransformer
preprocessor_instance = PreprocessingTransformer()

# Load and preprocess data
tips_data = load_tips_data()  # Assuming load_tips_data is defined elsewhere
X = tips_data.drop('tip', axis=1)
y = (tips_data['tip'] > 5).astype(int)

# Fit the preprocessor & Transform the data
preprocessor_instance.fit(X, y)
transformed_data = preprocessor_instance.transform(X)

# Print the first few rows of the transformed data
print(transformed_data[:5])
