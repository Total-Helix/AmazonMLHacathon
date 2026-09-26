import os
import zipfile
from datetime import datetime

def package_submission():
    print("=== Amazon ML Challenge - Final Zipping ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"TeamName_submission_{timestamp}.zip"
    
    # Tanuj's new structure
    folders_to_zip = [
        'output/matching_results.tsv',
        'output/candidate_pairs.tsv',
        'src',
        'main.py',
        'tanuj_requirements.txt',
        'Documentation_template.md'
    ]
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in folders_to_zip:
            if not os.path.exists(item):
                print(f"Missing required folder/file: {item}")
                continue
                
            if os.path.isfile(item):
                zipf.write(item, item)
            else:
                for root, dirs, files in os.walk(item):
                    for file in files:
                        if '__pycache__' not in root and not file.endswith('.pyc'):
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, file_path)
                            
    print(f"\nSuccess! Ready for upload: {zip_filename}")
    print("Upload 'output/matching_results.tsv' to the portal and keep the zip for final review.")

if __name__ == "__main__":
    package_submission()
