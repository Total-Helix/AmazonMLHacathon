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
        'role1_gpu_preprocessor.py',
        'role2_faiss_blocking.py',
        'MASTER_MENU.bat',
        'requirements.txt',
        'Documentation_template.md'
    ]
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in folders_to_zip:
            if not os.path.exists(item):
                print(f"Missing required folder/file: {item}")
                continue
                
            if os.path.isfile(item):
                # If it's an output file or documentation, keep it at root level (or output/)
                if item.startswith('output/') or item == 'Documentation_template.md':
                    zip_path = item
                else:
                    # Put code files inside the required code/business_entity_resolution/ folder
                    zip_path = f"code/business_entity_resolution/{item}"
                zipf.write(item, zip_path)
            else:
                for root, dirs, files in os.walk(item):
                    for file in files:
                        if '__pycache__' not in root and not file.endswith('.pyc'):
                            file_path = os.path.join(root, file)
                            
                            # Keep output/ in output/
                            if file_path.startswith('output\\') or file_path.startswith('output/'):
                                zip_path = file_path
                            else:
                                # Put all other code inside the required folder
                                zip_path = f"code/business_entity_resolution/{file_path}"
                                
                            zipf.write(file_path, zip_path)
                            
    print(f"\nSuccess! Ready for upload: {zip_filename}")
    print("-> 1. Upload 'output/matching_results.tsv' to the portal leaderboard.")
    print("-> 2. Keep the zip file for the Final Submission Package.")

if __name__ == "__main__":
    package_submission()
