from fastapi import HTTPException
import numpy as np

from apt.utils.datasets import ArrayDataset
from apt.anonymization import Anonymize
from diffprivlib.mechanisms import Laplace, Gaussian

from utils.enums import DatasetEnum, MitigationEnum

def apply_mitigation(dataset_id, preprocessed_dataset, art_classifier, mitigation_technique):
  match dataset_id:
    case DatasetEnum.ADULT_CENSUS:
      features_to_be_processed = [0, 1, 2, 4]
    
    case DatasetEnum.NURSERY:
      features_to_be_processed = [5, 6, 7]
    
    case DatasetEnum.GERMAN_CREDIT_SCORING:
      features_to_be_processed = [1, 2, 3, 9, 11, 13, 14, 16]
    
    case _:
      raise HTTPException(status_code=400, detail=f"Dataset id {dataset_id} not found")
    
  match mitigation_technique:
    case MitigationEnum.ANONYMIZATION:
      result = anonymization(preprocessed_dataset, features_to_be_processed, art_classifier)
    
    case MitigationEnum.DIFFERENTIAL_PRIVACY:
      result = differential_privacy(preprocessed_dataset, features_to_be_processed)
    
    case _:
      raise HTTPException(status_code=400, detail=f"Mitigation Technique {mitigation_technique} not found")
    
  return result
    

def anonymization(preprocessed_dataset, QI, art_classifier):
  (x_train, y_train), (x_test, y_test) = preprocessed_dataset
    
  x_train_predictions = np.array([np.argmax(arr) for arr in art_classifier.predict(x_train)])
  anonymizer = Anonymize(100, QI)
  x_train = anonymizer.anonymize(ArrayDataset(x_train, x_train_predictions))
    
  return (x_train, y_train), (x_test, y_test)

def differential_privacy(preprocessed_dataset, sensitive_columns, epsilon=0.5, dp_type="laplace"):
  (x_train, y_train), (x_test, y_test) = preprocessed_dataset
  dp_x_train = np.copy(x_train)

  for col in sensitive_columns:
      column_data = x_train[:, col]
      sensitivity = np.max(column_data) - np.min(column_data)  # Sensitivity estimation
      
      if dp_type == "laplace":
        dp_mechanism = Laplace(epsilon=epsilon, sensitivity=sensitivity)
      elif dp_type == "gaussian":
        delta = 1e-5  # Small delta for Gaussian mechanism
        dp_mechanism = Gaussian(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
      else:
        raise ValueError("Invalid dp_type. Choose 'laplace' or 'gaussian'.")

      # Apply DP noise to each value
      dp_x_train[:, col] = [dp_mechanism.randomise(val) for val in column_data]

  return (dp_x_train, y_train), (x_test, y_test)