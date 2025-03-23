from fastapi import FastAPI, HTTPException, BackgroundTasks
import pandas as pd
import numpy as np
import uvicorn
import os
import json
import time
import logging
from datetime import datetime

from utils.enums import *
from utils.dataset_loader import load_dataset
from utils.model_trainer import train_model
from utils.attack_runner import run_attack
from utils.mitigation_applier import apply_mitigation

from initialize import initialize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Model Training and Evaluation API")

initialize()

# At the top of your file, add a global state tracker
active_sessions = {}

class SessionState:
    def __init__(self, session_id):
        self.session_id = session_id
        self.dataset_id = None
        self.dataset = None
        self.model_type = None
        self.model = None
        self.art_classifier = None
        self.preprocessed_dataset = None
        self.training_result = None
        self.attack_type = None
        self.attack_result = None
        
        self.mitigation_technique = None
        self.mitigated_dataset = None
        self.mitigated_art_classifier = None
        self.mitigated_model = None
        self.mitigated_training_result = None
        self.mitigated_attack_result = None

@app.post("/api/start-session", response_model=dict)
def start_session():
    """Start a new analysis session"""
    session_id = f"session_{int(time.time())}"
    active_sessions[session_id] = SessionState(session_id)
    return {"session_id": session_id, "message": "Session started successfully"}

@app.post("/api/load-dataset", response_model=dict)
def load_dataset_endpoint(session_id: str, dataset_id: DatasetEnum):
    """Load a dataset for the given session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = active_sessions[session_id]
    
    # Load the dataset
    try:
        dataset = load_dataset(dataset_id)                        
        session.dataset_id = dataset_id
        session.dataset = dataset
        
        return {
            "session_id": session_id,
            "dataset_id": dataset_id,
            "message": "Dataset loaded successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train-model", response_model=dict)
def train_model_endpoint(session_id: str, model_type: ModelEnum):
    """Train a model on the loaded dataset"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = active_sessions[session_id]
    
    # Verify dataset is loaded
    if session.dataset is None:
        raise HTTPException(status_code=400, detail="Dataset must be loaded before training a model")
    
    # Train the model
    try:
        model, base_model_accuracy, art_classifier, preprocessed_dataset = train_model(session.dataset_id, session.dataset, model_type)
        session.model_type = model_type
        session.model = model
        session.training_result = base_model_accuracy
        session.art_classifier = art_classifier
        session.preprocessed_dataset = preprocessed_dataset
        
        return {
            "session_id": session_id,
            "message": "Model trained successfully",
            "model_type": model_type,
            "base_model_accuracy": base_model_accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/run-attack", response_model=dict)
def run_attack_endpoint(session_id: str, attack_type: AttackEnum):
    """Run an attack on the trained model"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = active_sessions[session_id]
    
    # Verify model is trained
    if session.model is None:
        raise HTTPException(status_code=400, detail="Model must be trained before running an attack")
    
    # Run the attack
    try:
        attack_result = run_attack(session.preprocessed_dataset, session.art_classifier, attack_type)
        session.attack_type = attack_type
        session.attack_result = attack_result
        
        return {
            "session_id": session_id,
            "message": "Attack executed successfully",
            "attack_type": attack_type,
            "attack_result": attack_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/apply-mitigation", response_model=dict)
def apply_mitigation_endpoint(session_id: str, mitigation_technique: MitigationEnum):
    """Apply mitigation technique, retrain model, and re-run attack"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = active_sessions[session_id]
    
    # Verify previous steps completed
    if session.dataset is None:
        raise HTTPException(status_code=400, detail="Dataset must be loaded first")
    if session.model is None:
        raise HTTPException(status_code=400, detail="Model must be trained first")
    if session.attack_result is None:
        raise HTTPException(status_code=400, detail="Attack must be run first")
    
    try:
        # Apply mitigation
        mitigated_dataset = apply_mitigation(session.dataset_id, session.preprocessed_dataset, session.art_classifier, mitigation_technique)
        session.mitigation_technique = mitigation_technique
        session.mitigated_dataset = mitigated_dataset
        
        # Retrain model on mitigated dataset
        mitigated_model, mitigated_training_result, mitigated_art_classifier, _ = train_model(session.dataset_id, mitigated_dataset, session.model_type, True)
        session.mitigated_model = mitigated_model
        session.mitigated_training_result = mitigated_training_result
        session.mitigated_art_classifier = mitigated_art_classifier

        # Re-run the attack on the retrained model
        mitigated_attack_result = run_attack(session.preprocessed_dataset, mitigated_art_classifier, session.attack_type)
        session.mitigated_attack_result = mitigated_attack_result
        
        return {
            "session_id": session_id,
            "message": "Mitigation applied, model retrained, and attack re-run successfully",
            "mitigation_technique": mitigation_technique,
            "original_training_result": session.training_result,
            "mitigated_training_result": mitigated_training_result,
            "original_attack_result": session.attack_result,
            "mitigated_attack_result": mitigated_attack_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/session-status/{session_id}")
# def get_session_status(session_id: str):
#     """Get the current status of a session"""
#     if session_id not in active_sessions:
#         raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
#     session = active_sessions[session_id]
    
#     # Determine the current stage
#     if session.mitigated_attack_result is not None:
#         current_stage = "mitigation_complete"
#     elif session.attack_result is not None:
#         current_stage = "attack_complete"
#     elif session.model is not None:
#         current_stage = "model_trained"
#     elif session.dataset is not None:
#         current_stage = "dataset_loaded"
#     else:
#         current_stage = "session_started"
    
#     return {
#         "session_id": session_id,
#         "current_stage": current_stage,
#         "dataset_id": session.dataset_id,
#         "model_type": session.model_type,
#         "attack_type": session.attack_type,
#         "mitigation_technique": session.mitigation_technique
#     }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
