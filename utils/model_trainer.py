from fastapi import HTTPException
from utils.model_preprocessor import preprocess_adult_census_using_inference, preprocess_adult_census_using_one_hot_encoder, preprocess_german_credit_scoring
from utils.enums import ModelEnum, DatasetEnum

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

def train_model(dataset_id, dataset, model_type, model_params):
  (x_train, y_train), (x_test, y_test) = dataset

  match dataset_id:
    case DatasetEnum.ADULT_CENSUS:
      (x_test, x_train), (y_test, y_train) = preprocess_adult_census_using_inference(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    case DatasetEnum.GERMAN_CREDIT_SCORING:
      (x_test, x_train), (y_test, y_train) = preprocess_german_credit_scoring(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
      
    case DatasetEnum.NURSERY:
      pass
    
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
  base_model_accuracy = model.score(x_test, y_test)
  
  return model, base_model_accuracy