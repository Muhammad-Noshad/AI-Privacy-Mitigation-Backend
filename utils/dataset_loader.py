from fastapi import HTTPException
from utils.enums import DatasetEnum
from apt.utils.dataset_utils import get_adult_dataset_pd, get_german_credit_dataset_pd, get_nursery_dataset_pd

def load_dataset(dataset_id):  
  match dataset_id:
    case DatasetEnum.ADULT_CENSUS:
      return get_adult_dataset_pd()
    
    case DatasetEnum.GERMAN_CREDIT_SCORING:
      return get_german_credit_dataset_pd()
    
    case DatasetEnum.NURSERY:
      return get_nursery_dataset_pd()
    
    case _:
      raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")