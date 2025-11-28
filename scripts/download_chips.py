import ee
import pandas as pd
import os
import requests
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 1. GOOGLE CLOUD PROJECT ID
GOOGLE_CLOUD_PROJECT = 'ai7102finalproject' 

# 2. Path to your CSV
CSV_FOLDER_PATH = '/Users/marinaananyan/ca-lulc/src/data'

# 3. Output Directory
OUTPUT_DIR = '../dataset_central_asia'

# 4. Asset Base Path
ASSET_BASE_PATH = 'users/marinaananyan/AI7102FinalProject/LULC_Stack_'

# 5. Chip Parameters
CHIP_SIZE_PX = 224
RESOLUTION = 30  
BUFFER_SIZE_M = (CHIP_SIZE_PX * RESOLUTION) / 2

# 6. COUNTRY LIST
COUNTRIES = ['Turkmenistan', 'Kazakhstan'] 


def create_session():
    """Creates a requests Session with automatic retries."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    return session

GLOBAL_SESSION = create_session()

def initialize_gee():
    try:
        ee.Initialize(
            project=GOOGLE_CLOUD_PROJECT,
            opt_url='https://earthengine-highvolume.googleapis.com'
        )
        print(f"GEE High-Volume initialized (Project: {GOOGLE_CLOUD_PROJECT}).")
    except Exception:
        print("Triggering authentication flow...")
        ee.Authenticate()
        ee.Initialize(
            project=GOOGLE_CLOUD_PROJECT,
            opt_url='https://earthengine-highvolume.googleapis.com'
        )

def download_single_chip(task_args):
    row, country_name, country_asset_path, output_folder = task_args
    
    try:
        # Parse Coordinates
        if isinstance(row['.geo'], str):
            geo_json = json.loads(row['.geo'])
            coords = geo_json['coordinates']
        else:
            coords = row['.geo']['coordinates']

        label_id = int(row['label'])
        sample_id = row.get('system:index', row.get('id', int(time.time()*100000)))
        
        save_dir = os.path.join(output_folder, country_name, f"Class_{label_id}")
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"chip_{sample_id}.tif"
        filepath = os.path.join(save_dir, filename)
        
        # Resume Check
        if os.path.exists(filepath):
            if os.path.getsize(filepath) > 0:
                return "Exists"
        
        # Define Request
        center_point = ee.Geometry.Point(coords)
        region = center_point.buffer(BUFFER_SIZE_M).bounds()
        image_asset = ee.Image(country_asset_path)
        
        # Get URL
        url = image_asset.getDownloadURL({
            'region': region,
            'format': 'GEO_TIFF',
            'dimensions': f'{CHIP_SIZE_PX}x{CHIP_SIZE_PX}'
        })
        
        response = GLOBAL_SESSION.get(url, timeout=30)
        
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return "Success"
        else:
            return f"HTTP Error {response.status_code}"
            
    except Exception as e:
        return f"Exception: {str(e)}"

def main():
    initialize_gee()
    
    print(f"\n STARTING TURBO DOWNLOAD FOR: {COUNTRIES} ")
    
    all_tasks = []
    for country in COUNTRIES:
        csv_filename = f"Samples_{country}.csv"
        csv_path = os.path.join(CSV_FOLDER_PATH, csv_filename)
        asset_path = f"{ASSET_BASE_PATH}{country}"
        
        if not os.path.exists(csv_path):
            print(f"Skipping {country}, CSV not found.")
            continue
            
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} points for {country}.")
        
        for _, row in df.iterrows():
            all_tasks.append((row, country, asset_path, OUTPUT_DIR))

    MAX_WORKERS = 12
    
    print(f"Starting Downloader with {MAX_WORKERS} threads...")
    
    results = {"Success": 0, "Exists": 0, "Error": 0}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_single_chip, task) for task in all_tasks]
        
        for future in tqdm(as_completed(futures), total=len(all_tasks), unit="chip"):
            res = future.result()
            if res == "Success": results["Success"] += 1
            elif res == "Exists": results["Exists"] += 1
            else: results["Error"] += 1

    print("\n--- DOWNLOAD COMPLETE ---")
    print(f"Downloaded: {results['Success']}")
    print(f"Skipped: {results['Exists']}")
    print(f"Errors: {results['Error']}")

if __name__ == "__main__":
    main()