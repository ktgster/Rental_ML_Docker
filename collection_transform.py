from credentials import mongo_db_cred
from pymongo import MongoClient
import pandas as pd
import numpy as np


def mongodb_to_dataframe(username, password, cluster_uri, db_name, collection_name):
    # Construct the MongoDB Atlas connection URI using the provided username and password
    mongo_uri = f"mongodb+srv://{username}:{password}@{cluster_uri}/{db_name}?retryWrites=true&w=majority"

    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Retrieve all documents from the collection
    cursor = collection.find()

    # Convert the cursor into a dataframe
    df = pd.DataFrame(list(cursor))

    # If the MongoDB documents have an '_id' field, it'll be added to the dataframe.
    # You can drop it if you don't want it in your dataframe.
    # if '_id' in df.columns:
    # df.drop('_id', axis=1, inplace=True)

    # Close the connection
    client.close()

    return df


def assign_quadrant(df):
    # Define the quadrant for each community
    NW_communities = [
        "Varsity",
        "Dalhousie",
        "Edgemont",
        "Brentwood",
        "Hamptons",
        "Arbour Lake",
        "Silver Springs",
        "University District",
        "University Heights",
        "Charleswood",
        "Collingwood",
        "Tuxedo Park",
        "Hidden Valley",
        "St Andrews Heights",
        "Nolan Hill",
        "Cambrian Heights",
        "North Haven",
        "Capitol Hill",
        "Ranchlands",
        "Hawkwood",
        "Bowness",
        "Rosedale",
        "Country Hills",
        "Royal Oak",
        "Citadel",
        "Sandstone",
        "Mount Pleasant",
        "Highland Park",
        "Montgomery",
        "Briar Hill",
        "Scenic Acres",
        "Collingwood",
        "Charleswood",
        "Silver Springs",
        "Dalhousie",
        "Hawkwood",
        "North Glenmore Park",
        "Rocky Ridge",
    ]

    NE_communities = [
        "Greenview",
        "Taradale",
        "Coventry Hills",
        "Marlborough",
        "Rundle",
        "Saddle Ridge",
        "Saddlebrook",
        "Redstone",
        "Cityscape",
        "Whitehorn",
        "Castleridge",
        "Pineridge",
        "Temple",
        "Falconridge",
        "Coral Springs",
        "Monterey Park",
        "Abbeydale",
        "Skyview",
        "Martindale",
        "Mayland Heights",
        "Vista Heights",
    ]

    SW_communities = [
        "Springbank Hill",
        "Bayview",
        "Richmond/Knob Hill",
        "Windsor Park",
        "Garrison Green",
        "Cougar Ridge",
        "Aspen Woods",
        "Erlton",
        "Bankview",
        "Westgate",
        "Lower Mount Royal",
        "Killarney",
        "Strathcona Park",
        "Altadore",
        "Cliff Bungalow",
        "Shaganappi",
        "Glamorgan",
        "Scarboro",
        "Mount Royal",
        "Rosscarrock",
        "Glendale",
        "Sunalta",
        "Wildwood",
        "Lincoln Park",
        "Evergreen",
        "Signal Hill",
        "Coach Hill",
        "Discovery Ridge",
        "Kingsland",
        "Haysboro",
        "Cedarbrae",
        "Glenbrook",
        "Rutland Park",
        "Lakeview",
        "Chinook Park",
        "Canyon Meadows",
        "Braeside",
        "Woodlands",
        "Belaire",
        "Palliser",
        "Pumphill",
        "Kelvin Grove",
    ]

    SE_communities = [
        "Downtown",
        "Victoria Park",
        "Beltline",
        "Acadia",
        "Bridgeland",
        "Mission",
        "Eau Claire",
        "Inglewood",
        "Forest Heights",
        "Albert Park",
        "Ogden",
        "Southview",
        "Dover",
        "Penbrooke Meadows",
        "Erin Woods",
        "New Brighton",
        "Forest Lawn",
        "Manchester",
        "Spruce Cliff",
        "Braeside",
        "Dover Glen",
        "Applewood",
        "Red Carpet",
        "Fonda",
        "Ramsay",
        "Radisson Heights",
        "Riverbend",
        "McKenzie Towne",
        "Copperfield",
        "Mckenzie Towne",
        "Douglas Glen",
        "Lake Bonavista",
        "Maple Ridge",
        "Queensland",
        "Lynnwood",
        "Elboya",
        "Deer Ridge",
        "McKenzie Lake",
        "Mahogany",
    ]

    Inner_city_communities = [
        "Connaught",
        "East Village",
        "Renfrew",
        "Crescent Heights",
        "South Calgary",
        "Evanston",
        "Huntington Hills",
        "Highwood",
        "Winston Heights",
        "Parkhill-Stanley Park",
        "Savanna",
        "Bridgeland",
        "Sunnyside",
        "Hillhurst",
        "West Springs",
        "Oakridge",
        "Shawnee Slopes",
        "Kincora",
        "Harvest Hills",
        "Sunnyside",
        "Tuxedo",
        "West Hillhurst",
        "Tuxedo Park",
        "Willow Park",
        "University Heights",
        "Sherwood",
        "Patterson",
        "Garrison Woods",
        "Greenwich",
        "Elbow Park",
        "Mayfair",
        "Country Hills Village",
        "Point McKay",
        "Nolan Hill",
        "Hotchkiss",
        "Montreux",
        "North Haven",
        "Beddington",
    ]

    Inner_city_communities = [
        "Connaught",
        "East Village",
        "Renfrew",
        "Crescent Heights",
        "South Calgary",
        "Evanston",
        "Huntington Hills",
        "Highwood",
        "Winston Heights",
        "Parkhill-Stanley Park",
        "Savanna",
        "Bridgeland",
        "Sunnyside",
        "Hillhurst",
        "West Springs",
        "Oakridge",
        "Shawnee Slopes",
        "Kincora",
        "Harvest Hills",
        "Sunnyside",
        "Tuxedo",
        "West Hillhurst",
        "Tuxedo Park",
        "Willow Park",
        "University Heights",
        "Sherwood",
        "Patterson",
        "Garrison Woods",
        "Greenwich",
        "Elbow Park",
        "Mayfair",
        "Country Hills Village",
        "Point McKay",
        "Nolan Hill",
        "Hotchkiss",
        "Montreux",
        "North Haven",
        "Beddington",
    ]

    # Create a new column named 'Quadrant'
    def assign_quadrant(community):
        if community in NW_communities:
            return "NW"
        elif community in NE_communities:
            return "NE"
        elif community in SW_communities:
            return "SW"
        elif community in SE_communities:
            return "SE"
        elif community in Inner_city_communities:
            return "Downtown"
        else:
            return "Unknown"  # for communities not listed

    df["Quadrant"] = df["community"].apply(assign_quadrant)
    return df


def data_cleaning(df, columns_to_keep, columns_to_drop_na):
    df = df[columns_to_keep]
    df.fillna("", inplace=True)
    df = df.dropna(subset=columns_to_drop_na)
    df = assign_quadrant(df)
    df = df[df["Quadrant"] != "Unknown"]
    # df['features'] = df['features'].str.replace(r"\['", "", regex=True)
    # df['features'] = df['features'].str.replace(r"\']", "", regex=True)
    # df['features'] = df['features'].str.replace(r"'", "", regex=True)
    # df['features'].replace(["", " ", "False", np.nan], ["", "", "", ""], inplace=True)
    # df['features'] = df['features'].str.replace(r"\['", "", regex=True)
    # df['utilities_included'] = df['utilities_included'].str.replace(r"\']", "", regex=True)
    # df['utilities_included'] = df['utilities_included'].str.replace(r"\['", "", regex=True)
    # df['utilities_included'] = df['utilities_included'].str.replace(r"'", "", regex=True)
    # df['utilities_included'] = df['utilities_included'].str.replace(r", See Full Description", "")
    # df['utilities_included'].replace(["", " ", "False", np.nan], ["", "", "", ""], inplace=True)
    df["baths_y"] = df["baths_y"].str.replace(r"\s*baths?\s*", "", case=False)
    df["beds"] = df["beds"].str.replace(r"\s*beds?\s*", "", case=False)
    df["sq_feet_y"] = df["sq_feet_y"].str.replace("[a-zA-Z]", "", regex=True)
    df["sq_feet_y"] = df["sq_feet_y"].str.replace(",", "")
    df["sq_feet_y"] = df["sq_feet_y"].str.replace(" ", "")
    pattern = r"^\d{3,}(\.\d+)?$"
    df = df[df["sq_feet_y"].str.match(pattern)]
    return df


def dataframe_to_mongodb(
    dataframe, username, password, cluster_uri, db_name, collection_name, partitions=4
):

    # Construct the MongoDB Atlas connection URI using the provided username and password
    mongo_uri = f"mongodb+srv://{username}:{password}@{cluster_uri}/{db_name}?retryWrites=true&w=majority"

    # Split the DataFrame into 'n' partitions
    df_partitions = np.array_split(dataframe, partitions)

    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    for partition in df_partitions:
        # Convert partition to dictionary format for MongoDB
        records = partition.to_dict(orient="records")

        # Insert records into the collection
        collection.insert_many(records)
        print(f"Inserted {len(records)} records into {db_name}.{collection_name}")

    # Close the connection
    client.close()


if __name__ == "__main__":
    df = mongodb_to_dataframe(
        username=mongo_db_cred["username"],
        password=mongo_db_cred["password"],
        cluster_uri=mongo_db_cred["cluster_uri"],
        db_name=mongo_db_cred["db_name"],
        collection_name=mongo_db_cred["collection_name_raw"],
    )
    # List of columns to retain
    columns_to_keep = [
        "type",
        "latitude",
        "longitude",
        "community",
        "cats",
        "dogs",
        "price_y",
        "baths_y",
        "sq_feet_y",
        "furnishing",
        "lease_term_y",
        "beds",
    ]
    columns_to_drop_na = [
        "cats",
        "dogs",
        "price_y",
        "baths_y",
        "sq_feet_y",
        "furnishing",
        "lease_term_y",
        "beds",
    ]
    df = data_cleaning(df, columns_to_keep, columns_to_drop_na)
    dataframe_to_mongodb(
        dataframe=df,
        username=mongo_db_cred["username"],
        password=mongo_db_cred["password"],
        cluster_uri=mongo_db_cred["cluster_uri"],
        db_name=mongo_db_cred["db_name"],
        collection_name=mongo_db_cred["collection_name_clean"],
        partitions=5,
    )
