from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split


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


def type_cast_columns(df, type_dict):
    """
    Type-cast specified columns in a DataFrame based on a given dictionary.

    Args:
    - df (pd.DataFrame): Input DataFrame.
    - type_dict (dict): Dictionary where keys are column names and values are target data types.

    Returns:
    - pd.DataFrame: DataFrame with specified columns type-cast.
    """
    for col, dtype in type_dict.items():
        df[col] = df[col].astype(dtype)
    return df


def remove_outliers(df, columns, lower_quantile=0.05, upper_quantile=0.95):
    """
    Removes outliers from specified columns in a DataFrame based on given quantiles.

    Args:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of column names for which outliers should be removed.
    - lower_quantile (float): Lower quantile for outlier definition. Default is 0.05.
    - upper_quantile (float): Upper quantile for outlier definition. Default is 0.95.

    Returns:
    - pd.DataFrame: DataFrame with outliers removed.
    """
    for col in columns:
        Q_lower = df[col].quantile(lower_quantile)
        Q_upper = df[col].quantile(upper_quantile)
        df = df[(df[col] >= Q_lower) & (df[col] <= Q_upper)]

    return df


def one_hot_encode(df, columns_to_encode):
    """
    One-hot encodes specified columns in a DataFrame and drops the original columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns_to_encode (list): List of column names to one-hot encode.

    Returns:
        pd.DataFrame: DataFrame with specified columns one-hot encoded and original columns dropped.
    """

    # One-hot encode specified columns and drop the original columns
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=False)

    return df_encoded


def split_data(df, label, test_size=0.2, random_state=None):
    """
    Split the DataFrame into training and testing sets.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing features and labels.
        label (str): The name of the label column in the DataFrame.
        test_size (float, optional): The proportion of the dataset to include in the test split (default is 0.2).
        random_state (int or None, optional): Seed for the random number generator (default is None).

    Returns:
        tuple: A tuple containing the following:
            - X_train (pd.DataFrame): Training data (features).
            - X_test (pd.DataFrame): Testing data (features).
            - y_train (pd.Series): Training data (labels).
            - y_test (pd.Series): Testing data (labels).
    """
    X = df.drop(columns=[label])  # Extract features
    y = df[label]  # Extract labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def standardize_columns(
    dataframe, columns_to_standardize_any=None, columns_to_min_max=None
):
    """
    Standardize specified columns in a DataFrame by subtracting the mean
    and dividing by the standard deviation.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        columns_to_standardize_any (list): List of column names to standardize (any values).
        columns_to_min_max (list): List of column names to standardize (min-max scaling).

    Returns:
        pd.DataFrame: A new DataFrame with specified columns standardized.
        dict: A dictionary containing mean, standard deviation, min, and max values for the specified columns.
    """
    # Copy the original DataFrame to avoid modifying the original data
    standardized_df = dataframe.copy()

    stats = {}

    if columns_to_standardize_any:
        # Iterate through each column to standardize (any values)
        for column in columns_to_standardize_any:
            # Convert column values to float (errors='coerce' will replace non-numeric values with NaN)
            standardized_df[column] = pd.to_numeric(
                standardized_df[column], errors="coerce"
            ).astype(float)

            # Calculate the mean and standard deviation
            mean = standardized_df[column].mean()
            std_dev = standardized_df[column].std()

            # Store stats
            stats[column] = {"mean": mean, "std_dev": std_dev}

            # Standardize the column
            standardized_df[column] = (standardized_df[column] - mean) / std_dev

    if columns_to_min_max:
        # Iterate through each column to standardize (min-max scaling)
        for column in columns_to_min_max:
            # Convert column values to float (errors='coerce' will replace non-numeric values with NaN)
            standardized_df[column] = pd.to_numeric(
                standardized_df[column], errors="coerce"
            ).astype(float)

            # Scale the positive values between 0 and 1 using min-max scaling
            min_value = standardized_df[column].min()
            max_value = standardized_df[column].max()

            # Store stats
            stats[column] = {"min": min_value, "max": max_value}

            standardized_df[column] = (standardized_df[column] - min_value) / (
                max_value - min_value
            )

    return standardized_df, stats
