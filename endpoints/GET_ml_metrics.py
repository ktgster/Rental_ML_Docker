from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import importlib.util

# Create an APIRouter instance
router = APIRouter()

# Defining allowed machine learning models in set for the API
ALLOWED_MODEL_TYPES = {
    "linear",
    "random_forest",
    "xgboost",
    "svr",
    "decision_tree",
    "gradient_boosting",
    "ridge",
    "lasso",
}

# Define pydantic data class for output
class Metrics(BaseModel):
    model_type: str
    metrics: dict


# Dynamic API endpoint for predictions
@router.get("/{model_type}/metrics/", response_model=Metrics)
async def get_model_metrics(model_type: str):
    try:
        if model_type not in ALLOWED_MODEL_TYPES:
            raise HTTPException(status_code=400, detail="Invalid model_type")

        # Specify the path to the Python file containing the metrics dictionary
        metrics_path = (
            Path(__file__).parent.parent / "models"
        ) / f"{model_type}_metrics.py"

        # Create a module object from the file
        metrics_module = importlib.util.spec_from_file_location("metrics", metrics_path)
        metrics = importlib.util.module_from_spec(metrics_module)

        # Load the module
        metrics_module.loader.exec_module(metrics)

        # Access the metrics dictionary from the imported module
        model_metrics = metrics.metrics

        result = {"model_type": model_type, "metrics": model_metrics}
        return result
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"An error occurred: {e}"},
        )
