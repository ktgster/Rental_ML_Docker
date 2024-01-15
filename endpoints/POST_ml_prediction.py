# Importing necessary libraries & modules
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import numpy as np
from stats import label_stats, feature_stats
from pathlib import Path

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

# define pydantic data class for output
class PredictionResponse(BaseModel):
    model_type: str
    prediction: float


# define pydantic data class for inputs
class HouseFeatures(BaseModel):
    baths_y: float
    sq_feet_y: float
    beds: float
    type_Apartment: float
    type_Apartment_Parking_Spot: float
    type_Apartment_Townhouse: float
    type_Basement: float
    type_Condo_Unit: float
    type_Condo_Unit_Apartment: float
    type_Duplex: float
    type_House: float
    type_Loft: float
    type_Main_Floor: float
    type_Room_For_Rent: float
    type_Townhouse: float
    type_Townhouse_Apartment: float
    community_Abbeydale: float
    community_Acadia: float
    community_Albert_Park: float
    community_Altadore: float
    community_Applewood: float
    community_Arbour_Lake: float
    community_Aspen_Woods: float
    community_Bankview: float
    community_Bayview: float
    community_Beddington: float
    community_Beltline: float
    community_Bowness: float
    community_Braeside: float
    community_Brentwood: float
    community_Briar_Hill: float
    community_Bridgeland: float
    community_Cambrian_Heights: float
    community_Canyon_Meadows: float
    community_Capitol_Hill: float
    community_Castleridge: float
    community_Cedarbrae: float
    community_Charleswood: float
    community_Chinook_Park: float
    community_Citadel: float
    community_Cityscape: float
    community_Cliff_Bungalow: float
    community_Coach_Hill: float
    community_Collingwood: float
    community_Connaught: float
    community_Copperfield: float
    community_Coral_Springs: float
    community_Cougar_Ridge: float
    community_Country_Hills: float
    community_Country_Hills_Village: float
    community_Coventry_Hills: float
    community_Crescent_Heights: float
    community_Dalhousie: float
    community_Deer_Ridge: float
    community_Discovery_Ridge: float
    community_Douglas_Glen: float
    community_Dover: float
    community_Dover_Glen: float
    community_Downtown: float
    community_East_Village: float
    community_Eau_Claire: float
    community_Edgemont: float
    community_Elboya: float
    community_Erin_Woods: float
    community_Erlton: float
    community_Evanston: float
    community_Evergreen: float
    community_Falconridge: float
    community_Fonda: float
    community_Forest_Heights: float
    community_Forest_Lawn: float
    community_Garrison_Green: float
    community_Garrison_Woods: float
    community_Glamorgan: float
    community_Glenbrook: float
    community_Glendale: float
    community_Greenview: float
    community_Greenwich: float
    community_Hamptons: float
    community_Harvest_Hills: float
    community_Hawkwood: float
    community_Haysboro: float
    community_Hidden_Valley: float
    community_Highland_Park: float
    community_Highwood: float
    community_Hillhurst: float
    community_Huntington_Hills: float
    community_Inglewood: float
    community_Kelvin_Grove: float
    community_Killarney: float
    community_Kincora: float
    community_Kingsland: float
    community_Lake_Bonavista: float
    community_Lakeview: float
    community_Lincoln_Park: float
    community_Lower_Mount_Royal: float
    community_Lynnwood: float
    community_Manchester: float
    community_Marlborough: float
    community_Martindale: float
    community_Mayland_Heights: float
    community_McKenzie_Towne: float
    community_Mckenzie_Towne: float
    community_Mission: float
    community_Monterey_Park: float
    community_Montgomery: float
    community_Montreux: float
    community_Mount_Pleasant: float
    community_Mount_Royal: float
    community_New_Brighton: float
    community_North_Glenmore_Park: float
    community_Oakridge: float
    community_Ogden: float
    community_Palliser: float
    community_Parkhill_Stanley_Park: float
    community_Patterson: float
    community_Penbrooke_Meadows: float
    community_Pineridge: float
    community_Point_McKay: float
    community_Queensland: float
    community_Radisson_Heights: float
    community_Ramsay: float
    community_Ranchlands: float
    community_Red_Carpet: float
    community_Redstone: float
    community_Renfrew: float
    community_Richmond_Knob_Hill: float
    community_Riverbend: float
    community_Rosedale: float
    community_Rosscarrock: float
    community_Royal_Oak: float
    community_Rundle: float
    community_Saddle_Ridge: float
    community_Saddlebrook: float
    community_Sandstone: float
    community_Savanna: float
    community_Scarboro: float
    community_Scenic_Acres: float
    community_Shaganappi: float
    community_Shawnee_Slopes: float
    community_Sherwood: float
    community_Signal_Hill: float
    community_Silver_Springs: float
    community_Skyview: float
    community_South_Calgary: float
    community_Southview: float
    community_Springbank_Hill: float
    community_Spruce_Cliff: float
    community_St_Andrews_Heights: float
    community_Strathcona_Park: float
    community_Sunalta: float
    community_Sunnyside: float
    community_Taradale: float
    community_Tuxedo: float
    community_Tuxedo_Park: float
    community_University_District: float
    community_University_Heights: float
    community_Varsity: float
    community_Victoria_Park: float
    community_Vista_Heights: float
    community_West_Hillhurst: float
    community_West_Springs: float
    community_Westgate: float
    community_Whitehorn: float
    community_Wildwood: float
    community_Willow_Park: float
    community_Windsor_Park: float
    community_Winston_Heights: float
    community_Woodlands: float
    cats_False: float
    cats_True: float
    dogs_False: float
    dogs_True: float
    lease_term_y_12_months: float
    lease_term_y_Long_Term: float
    lease_term_y_Negotiable: float
    lease_term_y_Short_Term: float
    Quadrant_Downtown: float
    Quadrant_NE: float
    Quadrant_NW: float
    Quadrant_SE: float
    Quadrant_SW: float


# Dynamic API endpoint for predictions
@router.post("/{model_type}/predict/", response_model=PredictionResponse)
async def predict_rent_price(model_type: str, input_data: HouseFeatures):
    try:
        if model_type not in ALLOWED_MODEL_TYPES:
            raise HTTPException(status_code=400, detail="Invalid model_type")
        # scaling
        input_data.sq_feet_y = (
            input_data.sq_feet_y - feature_stats["sq_feet_y"]["min"]
        ) / (feature_stats["sq_feet_y"]["max"] - feature_stats["sq_feet_y"]["min"])

        # create input_array based on pydantic dataclass
        input_array = np.array(list(input_data.dict().values())).reshape(1, -1)

        # Load your pre-trained model and make predictions
        model_path = (Path(__file__).parent.parent / "models") / f"{model_type}.pkl"
        with open(model_path, "rb") as model_file:
            data = pickle.load(model_file)
            loaded_model = data["model"]

        # Predict on input data
        prediction = loaded_model.predict(input_array)

        # Unscale
        prediction[0] = (
            prediction[0]
            * (label_stats["price_y"]["max"] - label_stats["price_y"]["min"])
        ) + label_stats["price_y"]["min"]

        # Return the prediction as a JSON response
        result = {"model_type": model_type, "prediction": float(prediction[0])}
        return result
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"An error occurred: {e}"},
        )
