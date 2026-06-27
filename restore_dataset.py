import pandas as pd
import urllib.request
import os

print("Downloading clean language dataset... please wait a moment.")
url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv"

try:
    # Safely download the multi-language CSV file
    urllib.request.urlretrieve(url, "language.csv")

    # Verify the file structure matches your project
    df = pd.read_csv("language.csv")
    print("\n--- DATASET RESTORED SUCCESSFULY ---")
    print(f"Total rows recovered: {len(df)}")
    print("Available languages in your new file:")
    print(df['language'].value_counts().head(10))
    print("-------------------------------------")

except Exception as e:
    print(f"Error downloading dataset: {e}")