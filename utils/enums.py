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
  DECISION_TREE = "decision_tree"
  LOGISTIC_REGRESSION = "logistic_regression"
  NAIVE_BAYES = "naive_bayes"
  NEURAL_NETWORK = "neural_network"
  # Add more models as needed


class AttackEnum(str, Enum):
  BLACK_BOX_ATTACK = "black_box_attack"
  WHITE_BOX_ATTACK = "white_box_attack"
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
