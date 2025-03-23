from fastapi import HTTPException
from utils.model_preprocessor import preprocess_adult_census_using_inference, preprocess_german_credit_scoring, preprocess_nursery
from utils.enums import ModelEnum, DatasetEnum

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from art.estimators.classification.scikitlearn import ScikitlearnGaussianNB
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression

def train_model(dataset_id, dataset, model_type, preprocessed=False):
  (x_train, y_train), (x_test, y_test) = dataset

  if not preprocessed:
    match dataset_id:
      case DatasetEnum.ADULT_CENSUS:
        (x_train, y_train), (x_test, y_test) = preprocess_adult_census_using_inference(x_train, y_train, x_test, y_test)
      
      case DatasetEnum.GERMAN_CREDIT_SCORING:
        (x_train, y_train), (x_test, y_test) = preprocess_german_credit_scoring(x_train, y_train, x_test, y_test)
        
      case DatasetEnum.NURSERY:
        (x_train, y_train), (x_test, y_test) = preprocess_nursery(x_train, y_train, x_test, y_test)
      
      case _:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
      
  
  match model_type:
    case ModelEnum.DECISION_TREE:
      model = DecisionTreeClassifier()
    
    case ModelEnum.LOGISTIC_REGRESSION:
      model = LogisticRegression()
        
    case ModelEnum.NAIVE_BAYES:
      model = GaussianNB()
    
    case _:
      raise HTTPException(status_code=404, detail=f"Model type {model_type} not supported")
  
  model.fit(x_train, y_train) 
  
  match model_type:
    case ModelEnum.DECISION_TREE:
      art_classifier = ScikitlearnDecisionTreeClassifier(model)
    
    case ModelEnum.LOGISTIC_REGRESSION:
      art_classifier = ScikitlearnLogisticRegression(model)
        
    case ModelEnum.NAIVE_BAYES:
      art_classifier = ScikitlearnGaussianNB(model)
    
    case _:
      raise HTTPException(status_code=404, detail=f"Model type {model_type} not supported")
  
  base_model_accuracy = model.score(x_test, y_test)
  
  preprocessed_dataset = (x_train, y_train), (x_test, y_test)
  
  return model, base_model_accuracy, art_classifier, preprocessed_dataset