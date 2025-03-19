import os

def initialize():
  create_directories()
  
def create_directories():
  # Define path to datasets and models
  DATASETS_DIR = "datasets/"
  MODELS_DIR = "models/"
  RESULTS_DIR = "results/"

  # Create directories if they don't exist
  os.makedirs(DATASETS_DIR, exist_ok=True)
  os.makedirs(MODELS_DIR, exist_ok=True)
  os.makedirs(RESULTS_DIR, exist_ok=True)