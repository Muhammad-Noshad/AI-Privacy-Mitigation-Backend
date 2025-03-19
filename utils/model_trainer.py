from utils.model_preprocessor import preprocess_adult_census_using_inference, preprocess_adult_census_using_one_hot_encoder
from utils.enums import DatasetEnum
from sklearn.tree import DecisionTreeClassifier
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier

def train_model(dataset, model_type, model_params):
  (x_train, y_train), (x_test, y_test) = dataset
    
  (x_test, x_train), (y_test, y_train) = preprocess_adult_census_using_inference(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
  
  model = DecisionTreeClassifier()
  model.fit(x_train, y_train)

  art_classifier = ScikitlearnDecisionTreeClassifier(model)

  base_model_accuracy = model.score(x_test, y_test)
  
  return model, base_model_accuracy