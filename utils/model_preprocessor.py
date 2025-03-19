def preprocess_adult_census(x_train, y_train, x_test, y_test):
  x_train = x_train.to_numpy()[:, [1, 3, 4, 5, 6, 7, 11]]
  y_train = y_train.to_numpy().astype(int)
  x_test = x_test.to_numpy()[:, [1, 3, 4, 5, 6, 7, 11]]
  y_test = y_test.to_numpy().astype(int)

  x_train = x_train[:x_test.shape[0]]
  y_train = y_train[:y_test.shape[0]]
  
