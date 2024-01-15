from feature_engineering import (
    mongodb_to_dataframe,
    type_cast_columns,
    remove_outliers,
    split_data,
    standardize_columns,
    one_hot_encode,
)
from credentials import mongo_db_cred
from models_and_metrics import train_and_predict


if __name__ == "__main__":
    # df from mongo db
    df = mongodb_to_dataframe(
        username=mongo_db_cred["username"],
        password=mongo_db_cred["password"],
        cluster_uri=mongo_db_cred["cluster_uri"],
        db_name=mongo_db_cred["db_name"],
        collection_name=mongo_db_cred["collection_name_clean"],
    )
    # keep certain columns
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
    # fix data
    df["beds"] = df["beds"].replace("Studio", "1")
    # type cast based on dictionary
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
    # cast data types
    df = type_cast_columns(df, type_dict)
    # Remove outliers based on quartile
    df = remove_outliers(
        df, columns=["price_y", "sq_feet_y"], lower_quantile=0.05, upper_quantile=0.95
    )
    # one hot encode categories
    df = one_hot_encode(
        df,
        columns_to_encode=[
            "type",
            "community",
            "cats",
            "dogs",
            "lease_term_y",
            "Quadrant",
        ],
    )
    # train test split
    X_train, X_test, y_train, y_test = split_data(
        df=df, label="price_y", test_size=0.2, random_state=123
    )
    X_train, feature_stats = standardize_columns(
        dataframe=X_train,
        columns_to_standardize_any=None,
        columns_to_min_max=["sq_feet_y"],
    )
    print(f"feature_stats: {feature_stats}")
    y_train, label_stats = standardize_columns(
        dataframe=y_train.to_frame(name="price_y"),
        columns_to_standardize_any=None,
        columns_to_min_max=["price_y"],
    )
    print(f"label_stats: {label_stats}")
    # Write dictionaries to a .py file
    with open("stats.py", "w") as f:
        f.write("feature_stats = " + str(feature_stats) + "\n")
        f.write("label_stats = " + str(label_stats) + "\n")

    train_and_predict(
        models_list=[
            "linear",
            "random_forest",
            "xgboost",
            "svr",
            "decision_tree",
            "gradient_boosting",
            "ridge",
            "lasso",
        ],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_stats=feature_stats,
        label_stats=label_stats,
    )
