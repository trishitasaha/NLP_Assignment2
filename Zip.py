import os
import zipfile
from tqdm import tqdm

folder_path = "Full"
zip_path = "Full.zip"

# Collect all files
all_files = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        all_files.append(os.path.join(root, file))

# Create ZIP with progress bar
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path in tqdm(all_files, desc="Zipping files", unit="file"):
        arcname = os.path.relpath(file_path, folder_path)
        zipf.write(file_path, arcname)

print(f"\nZIP file created successfully: {zip_path}")