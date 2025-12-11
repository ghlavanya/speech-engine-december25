import pandas as pd
from pathlib import Path
import zipfile

zip_path = Path(r"C:\Users\lavan\Downloads\archive.zip")
extract_to = Path(r"C:\Users\lavan\Downloads\archive")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Done extracting!")

#download_path = Path(r"")