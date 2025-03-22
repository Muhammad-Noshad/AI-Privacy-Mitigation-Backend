from fastapi import HTTPException
from apt.utils.datasets import ArrayDataset
from apt.anonymization import Anonymize
from utils.enums import DatasetEnum
from utils.model_preprocessor import preprocess_nursery, preprocess_german_credit_scoring
import numpy as np

def apply_mitigation(dataset_id, dataset, preprocessed_dataset, art_classifier, mitigation_technique, mitigation_params):
  (x_train, y_train), (x_test, y_test) = preprocessed_dataset
  (raw_x_train, raw_y_train), (raw_x_test, raw_y_test) = dataset
  
  match dataset_id:
    case DatasetEnum.ADULT_CENSUS:
      x_train_predictions = np.array([np.argmax(arr) for arr in art_classifier.predict(x_train)])
      QI = [0, 1, 2, 4]
      anonymizer = Anonymize(100, QI)
      anon = anonymizer.anonymize(ArrayDataset(x_train, x_train_predictions))
      x_train = anon
    
    case DatasetEnum.NURSERY:
      x_train_predictions = np.array([np.argmax(arr) for arr in art_classifier.predict(x_train)]).reshape(-1,1)
      categorical_features = ['children', 'parents', 'has_nurs', 'form', 'housing', 'finance', 'health']
      QI = ["finance", "social", "health"]
      anonymizer = Anonymize(100, QI, categorical_features=categorical_features)
      anon = anonymizer.anonymize(ArrayDataset(raw_x_train, x_train_predictions))
      (x_train, y_train), (x_test, y_test) = preprocess_nursery(anon, y_train, raw_x_test, y_test)
    
    case DatasetEnum.GERMAN_CREDIT_SCORING:
      x_train_predictions = np.array([np.argmax(arr) for arr in art_classifier.predict(x_train)])
      categorical_features = ["Existing_checking_account", "Credit_history", "Purpose", "Savings_account", "Present_employment_since", "Personal_status_sex", "debtors", "Property", "Other_installment_plans", "Housing", "Job"]
      QI = ["Duration_in_month", "Credit_history", "Purpose", "debtors", "Property", "Other_installment_plans", "Housing", "Job"]
      anonymizer = Anonymize(100, QI, categorical_features=categorical_features)
      anon = anonymizer.anonymize(ArrayDataset(raw_x_train, x_train_predictions))
      (x_train, y_train), (x_test, y_test) = preprocess_german_credit_scoring(anon, y_train, raw_x_test, y_test)
    
    case _:
      raise HTTPException(status_code=400, detail=f"Dataset id {dataset_id} not found")
    
  return (x_train, y_train), (x_test, y_test)