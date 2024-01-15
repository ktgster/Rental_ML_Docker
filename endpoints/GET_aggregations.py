from feature_engineering import mongodb_to_dataframe, type_cast_columns, remove_outliers
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from credentials import mongo_db_cred

# Initialize the APIRouter for route registration
router = APIRouter()

# Define allowed categories for aggregation
ALLOWED_CATEGORIES = {"type", "community", "cats", "dogs", "furnishing", "lease term y"}

# Define allowed aggregation methods
ALLOWED_AGGREGATION_TYPES = {
    "mean",
    "median",
    "sum",
    "min",
    "max",
    "std",
    "var",
    "count",
}

# Define numeric columns on which aggregation can be performed
ALLOWED_NUMERICS = {"price_y", "baths_y", "sq_feet_y"}

# Define pydantic model for structured response
class Aggregations(BaseModel):
    aggregation: dict


# API endpoint to fetch aggregated data based on specified parameters
@router.get(
    "/aggregations/{aggregation_type}/{category_column}/{numeric_column}",
    response_model=Aggregations,
)
def aggregate_data(aggregation_type: str, numeric_column: str, category_column: str):
    # Validate that provided aggregation method is supported
    if aggregation_type not in ALLOWED_AGGREGATION_TYPES:
        raise HTTPException(status_code=400, detail="Invalid aggregation type")

    # Validate that provided numeric column is supported
    if numeric_column not in ALLOWED_NUMERICS:
        raise HTTPException(status_code=400, detail="Invalid numeric column")

    # Validate that provided category column is supported
    if category_column and category_column not in ALLOWED_CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid category column")

    try:
        # Fetch data from MongoDB
        df = mongodb_to_dataframe(
            username=mongo_db_cred["username"],
            password=mongo_db_cred["password"],
            cluster_uri=mongo_db_cred["cluster_uri"],
            db_name=mongo_db_cred["db_name"],
            collection_name=mongo_db_cred["collection_name_clean"],
        )

        # Retain specific columns of interest from the dataframe
        df = df[
            [
                "type",
                "community",
                "cats",
                "dogs",
                "price_y",
                "baths_y",
                "sq_feet_y",
                "lease_term_y",
                "beds",
                "Quadrant",
            ]
        ]

        # Convert 'Studio' in 'beds' column to '1' for consistency
        df["beds"] = df["beds"].replace("Studio", "1")

        # Define dictionary for type casting columns
        type_dict = {
            "type": str,
            "community": str,
            "cats": bool,
            "dogs": bool,
            "price_y": float,
            "baths_y": float,
            "sq_feet_y": float,
            "lease_term_y": str,
            "beds": float,
            "Quadrant": str,
        }

        # Cast dataframe columns to appropriate data types
        df = type_cast_columns(df, type_dict)

        # Remove outliers from specified columns based on quantiles
        df = remove_outliers(
            df,
            columns=["price_y", "sq_feet_y"],
            lower_quantile=0.05,
            upper_quantile=0.95,
        )

        # Perform the required aggregation
        grouped = df.groupby(category_column)[numeric_column].agg(aggregation_type)

        # Return aggregated results as a dictionary
        return {"aggregation": grouped.to_dict()}

    except Exception as e:
        # Handle any unforeseen errors and return a structured error message
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"An error occurred: {e}"},
        )
