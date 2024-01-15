from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time
import re
import json
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from credentials import chrome_cred, mongo_db_cred


def get_driver(url, chrome_driver_path, sleep_time=5):
    """
    Initialize a Chrome web driver using the specified driver path and navigate to the given URL.

    Parameters:
    - url: The target URL to open.
    - chrome_driver_path: The path to the ChromeDriver executable.
    - sleep_time: Optional. Time in seconds to wait after opening the URL to allow for page load. Defaults to 5 seconds.

    Returns:
    - An instance of the Chrome web driver opened to the given URL.
    """

    service = Service(executable_path=chrome_driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    time.sleep(sleep_time)
    return driver


def get_url_list(driver, pattern, sleep_time=5):
    """
    Scrape and collect URLs from a paginated source using the provided Selenium driver and a regex pattern.

    Parameters:
    - driver: The Selenium WebDriver instance to be used for scraping.
    - pattern (str): A regex pattern to match URLs in the page's HTML content.
    - sleep_time (int, optional): Time in seconds to wait between scraping operations. Defaults to 5 seconds.

    Returns:
    - list: A list of collected URLs.

    Note:
    - The scraping process is based on specific HTML structures and patterns, which may change if the source website updates its design.
    - The function assumes that the source website's pagination is based on the presence of certain buttons and uses the 'https://www.rentfaster.ca/' base URL for the matched paths.
    """
    url_list = []
    i = 1
    try:
        while True:
            time.sleep(sleep_time)
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, "html.parser")
            matches = re.findall(pattern, html_content)
            if len(matches) > 0:
                print(f"There are matches! for page {i}")
            else:
                print(f"No matches found. for page {i}")
            matches = list(set(matches))
            matches = [s.replace('href="', "").replace('"', "") for s in matches]
            matches = ["https://www.rentfaster.ca/" + s for s in matches]
            url_list.extend(matches)
            i = i + 1
            button = driver.find_element(
                By.XPATH,
                f'//a[@class="button is-rounded is-round ng-binding ng-scope" and text()="{i}"]',
            )
            button.click()
    except NoSuchElementException:
        pass
    return url_list


def scrape(driver, url_list, sleep_time=3):
    """
    Scrape structured data from a list of URLs using Selenium and BeautifulSoup.

    The function navigates to each URL in the provided list, extracts specific JavaScript content,
    and parses this content to accumulate structured data into a pandas DataFrame.

    Parameters:
    - driver: The Selenium WebDriver instance to be used for scraping.
    - url_list (list): A list of URLs to scrape.
    - sleep_time (int, optional): Time in seconds to wait after navigating to a URL. Defaults to 3 seconds.

    Returns:
    - DataFrame: A pandas DataFrame containing the scraped and structured data from the provided URLs.

    Note:
    - The scraping and parsing processes are based on specific HTML structures and patterns, which may change if the source websites update their designs.
    - The function expects the presence of specific JavaScript patterns ('var listingJson' and 'window.units') within the scraped content.
    - Each row in the final DataFrame is associated with a unique_id which is incrementally generated for each URL.
    """
    df_final = pd.DataFrame()
    z = 0
    for k in url_list:
        try:
            driver.get(k)
            time.sleep(sleep_time)
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, "html.parser")
            # print(soup.prettify())
            pattern = re.compile(r"<script>(.*?)</script>", re.DOTALL)
            matches = pattern.findall(html_content)
            matches = [
                element
                for element in matches
                if "var listingJson" in element or "window.units" in element
            ]

            listingJson_dict = {}
            window_listing_dict = {}
            units_array = []

            for data_string in matches:
                # Extracting content between curly braces for dictionaries
                pattern_1 = re.compile(r"{(.*?)}")
                matches_1 = pattern_1.findall(data_string)

                # Extracting content between square brackets for arrays
                pattern_2 = re.compile(r"window.units = (\[.*?\]);")
                match_2 = pattern_2.search(data_string)

                # If there are two dictionary matches, load them
                if len(matches_1) == 2:
                    listingJson_dict = json.loads("{" + matches_1[0] + "}")
                    window_listing_dict = json.loads("{" + matches_1[1] + "}")
                elif match_2:
                    units_array_str = match_2.group(1)
                    units_array = json.loads(units_array_str)
            df_1 = pd.DataFrame([listingJson_dict])
            df_2 = pd.DataFrame([window_listing_dict])
            df_3 = pd.DataFrame()
            df_1["unique_id"] = z
            for i in units_array:
                temp_df = pd.DataFrame(
                    [i]
                )  # Convert the dictionary to single-row dataframe
                df_3 = pd.concat([df_3, temp_df], ignore_index=True)
            df_1["unique_id"] = z
            df_3["unique_id"] = z
            z = z + 1
            df_4 = pd.merge(df_1, df_3, on="unique_id", how="outer")
            df_final = pd.concat([df_final, df_4], ignore_index=True)
            print(f"Successful at url {k}")
            print(df_4.iloc[:, :4])
        except Exception as e:
            print(f"Error at url {k}: {e}")
            continue
    df_final["sysdate"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return df_final


def dataframe_to_mongodb(
    dataframe, username, password, cluster_uri, db_name, collection_name, partitions=1
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
    print(chrome_cred["chrome_driver_path"])
    try:
        driver = get_driver(
            url="https://www.rentfaster.ca/ab/calgary/rentals/?l=11,51.0458,-114.0575&type=House&type=Townhouse&type=Loft&type=Condo%20Unit&type=Apartment&type=Duplex&type=Main%20Floor&type=Room%20For%20Rent&type=Basement&type=Mobile&type=Vacation%20Home&type=Storage&type=Office%20Space&type=Parking%20Spot&type=Acreage&limit=2895,3523#dialog-listview",
            chrome_driver_path=chrome_cred["chrome_driver_path"],
            sleep_time=5,
        )

        if not driver:
            raise Exception("Failed to initialize the web driver.")

        url_list = get_url_list(
            driver, pattern=r'href="\/ab\/calgary\/rentals\/.*?"', sleep_time=5
        )
        if not url_list:
            raise Exception("Failed to retrieve URL list.")

        df = scrape(driver, url_list, sleep_time=3)
        if df.empty:
            raise Exception("Scraping returned an empty DataFrame.")

        dataframe_to_mongodb(
            dataframe=df,
            username=mongo_db_cred["username"],
            password=mongo_db_cred["password"],
            cluster_uri=mongo_db_cred["cluster_uri"],
            db_name=mongo_db_cred["db_name"],
            collection_name=mongo_db_cred["collection_name"],
            partitions=1,
        )

    except Exception as e:
        print(f"An error occurred: {e}")
