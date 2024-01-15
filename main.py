# Importing necessary libraries & modules
from fastapi import FastAPI
from endpoints.POST_ml_prediction import router as predictions_app
from endpoints.GET_ml_metrics import router as metrics_app
from endpoints.GET_aggregations import router as aggregations_app

# Initializing FastAPI application
app = FastAPI()

# Mount the prediction app routes to the main app
app.include_router(predictions_app)
app.include_router(metrics_app)
app.include_router(aggregations_app)

# Simple health check endpoint
@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
