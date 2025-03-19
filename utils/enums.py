from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Define Enums for available options
class DatasetEnum(str, Enum):
  ADULT_CENSUS = "adult_census"
  GERMAN_CREDIT_SCORING = "german_credit_scoring"
  NURSERY = "nursery"
  # Add more datasets as needed


class ModelEnum(str, Enum):
  LOGISTIC_REGRESSION = "logistic_regression"
  RANDOM_FOREST = "random_forest"
  NEURAL_NETWORK = "neural_network"
  # Add more models as needed


class AttackEnum(str, Enum):
  MEMBERSHIP_INFERENCE = "membership_inference"
  MODEL_INVERSION = "model_inversion"
  ATTRIBUTE_INFERENCE = "attribute_inference"
  # Add more attacks as needed


class MitigationEnum(str, Enum):
  ANONYMIZATION = "anonymization"
  DIFFERENTIAL_PRIVACY = "differential_privacy"
  FEATURE_REMOVAL = "feature_removal"
  # Add more mitigation techniques as needed


# Define request and response models
class ModelTrainingRequest(BaseModel):
  dataset_id: DatasetEnum
  model_type: ModelEnum
  preprocessing_options: Optional[Dict[str, Any]] = None
  model_params: Optional[Dict[str, Any]] = None


class AttackRequest(BaseModel):
  job_id: str
  attacks: List[AttackEnum]


class MitigationRequest(BaseModel):
  job_id: str
  mitigation_technique: MitigationEnum
  mitigation_params: Optional[Dict[str, Any]] = None


class JobStatus(BaseModel):
  job_id: str
  status: str
  progress: float = 0.0
  result_url: Optional[str] = None
  error: Optional[str] = None
