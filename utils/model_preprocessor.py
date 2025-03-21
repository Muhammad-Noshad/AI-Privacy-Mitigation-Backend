import scipy
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def preprocess_adult_census_using_inference(x_train, y_train, x_test, y_test):
  x_train = x_train.to_numpy()
  y_train = y_train.to_numpy().astype(int)
  x_test = x_test.to_numpy()
  y_test = y_test.to_numpy().astype(int)

  x_train = x_train[:, [0, 2, 8, 9, 10]].astype(int)
  x_test = x_test[:, [0, 2, 8, 9, 10]].astype(int)

  x_train = x_train[:x_test.shape[0]]
  y_train = y_train[:y_test.shape[0]]
  
  return (x_train, y_train), (x_test, y_test)

  
def preprocess_adult_census_using_one_hot_encoder(x_train, y_train, x_test, y_test):
  x_train = x_train.to_numpy()[:, [1, 3, 4, 5, 6, 7, 11]]
  y_train = y_train.to_numpy().astype(int)
  x_test = x_test.to_numpy()[:, [1, 3, 4, 5, 6, 7, 11]]
  y_test = y_test.to_numpy().astype(int)

  x_train = x_train[:x_test.shape[0]]
  y_train = y_train[:y_test.shape[0]]

  preprocessor = OneHotEncoder(handle_unknown="ignore")

  x_train = preprocessor.fit_transform(x_train)
  x_test = preprocessor.transform(x_test)
  
  if scipy.sparse.issparse(x_train):
    x_train = x_train.toarray().astype(int)
    
  if scipy.sparse.issparse(x_test):
    x_test = x_test.toarray().astype(int)
  
  return (x_train, y_train), (x_test, y_test)


def preprocess_german_credit_scoring(x_train, y_train, x_test, y_test):
  features = ["Existing_checking_account", "Duration_in_month", "Credit_history", "Purpose", "Credit_amount", "Savings_account", "Present_employment_since", "Installment_rate", "Personal_status_sex", "debtors", "Present_residence", "Property", "Age", "Other_installment_plans", "Housing", "Number_of_existing_credits", "Job", "N_people_being_liable_provide_maintenance", "Telephone", "Foreign_worker"]
  categorical_features = ["Existing_checking_account", "Credit_history", "Purpose", "Savings_account", "Present_employment_since", "Personal_status_sex", "debtors", "Property", "Other_installment_plans", "Housing", "Job"]
  QI = ["Duration_in_month", "Credit_history", "Purpose", "debtors", "Property", "Other_installment_plans", "Housing", "Job"]

  numeric_features = [f for f in features if f not in categorical_features]
  numeric_transformer = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
  )
  categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
  preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
  )
  
  encoded_train = preprocessor.fit_transform(x_train)
  encoded_test = preprocessor.transform(x_test)  
  
  return (encoded_train, y_train), (encoded_test, y_test)