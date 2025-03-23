from fastapi import HTTPException
import numpy as np

from apt.utils.datasets import ArrayDataset
from apt.anonymization import Anonymize

from utils.enums import DatasetEnum

def apply_mitigation(dataset_id, preprocessed_dataset, art_classifier, mitigation_technique):
  (x_train, y_train), (x_test, y_test) = preprocessed_dataset
  
  match dataset_id:
    case DatasetEnum.ADULT_CENSUS:
      QI = [0, 1, 2, 4]
    
    case DatasetEnum.NURSERY:
      QI = [5, 6, 7]
    
    case DatasetEnum.GERMAN_CREDIT_SCORING:
      QI = [1, 2, 3, 9, 11, 13, 14, 16]
    
    case _:
      raise HTTPException(status_code=400, detail=f"Dataset id {dataset_id} not found")
    
  x_train_predictions = np.array([np.argmax(arr) for arr in art_classifier.predict(x_train)])
  anonymizer = Anonymize(100, QI)
  x_train = anonymizer.anonymize(ArrayDataset(x_train, x_train_predictions))
    
  return (x_train, y_train), (x_test, y_test)