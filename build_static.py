
import json
import shutil
import os
from app import process_data

def build():
    """Build local static files for GitHub Pages."""
    print("Fetching and processing data...")
    # Get the processed data structure
    data = process_data()
    
    # Save the JSON data
    print("Saving data.json...")
    with open('data.json', 'w') as f:
        json.dump(data, f)
        
    # Copy index.html to root
    # Note: app looks for templates/index.html, but GitHub pages will serve index.html from root
    print("Copying index.html...")
    shutil.copy('templates/index.html', 'index.html')
    
    print("Build complete. Files ready for GitHub Pages: index.html, data.json, static/")

if __name__ == "__main__":
    build()
