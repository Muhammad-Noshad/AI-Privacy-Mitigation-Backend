from sklearn.preprocessing import OneHotEncoder
import scipy

def preprocess_adult_census_using_inference(x_train, y_train, x_test, y_test):
  x_train = x_train.to_numpy()
  y_train = y_train.to_numpy().astype(int)
  x_test = x_test.to_numpy()
  y_test = y_test.to_numpy().astype(int)

  x_train = x_train[:, [0, 2, 8, 9, 10]].astype(int)
  x_test = x_test[:, [0, 2, 8, 9, 10]].astype(int)

  x_train = x_train[:x_test.shape[0]]
  y_train = y_train[:y_test.shape[0]]
  
  return (x_test, x_train), (y_test, y_train)

  
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
  
  return (x_test, x_train), (y_test, y_train)