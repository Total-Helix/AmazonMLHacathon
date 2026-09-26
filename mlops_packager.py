import os
import subprocess
import zipfile
from datetime import datetime

def package_submission():
    print("=== Amazon ML Challenge - MLOps Packager ===")
    
    # 1. Run the official validator first!
    print("\n[1/2] Running Validation Script...")
    validator_path = os.path.join('student_resource', 'utils', 'validate_submission.py')
    
    # If the user hasn't extracted the zip yet, warn them
    if not os.path.exists(validator_path):
        print(f"WARNING: Validator not found at {validator_path}.")
        print("Please extract '6ab10eb3b23ba_student_resource.zip' into a 'student_resource' folder.")
        print("Skipping validation for now... (DO NOT SKIP FOR FINAL SUBMISSION)\n")
    else:
        try:
            result = subprocess.run([
                'python', validator_path,
                '--matching', 'output/matching_results.tsv',
                '--candidate', 'output/candidate_pairs.tsv',
                '--test-dir', 'dataset/test'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print("VALIDATION FAILED! Fix these issues before submitting:")
                print(result.stdout)
                print(result.stderr)
                return # Stop the packaging process
            else:
                print("VALIDATION PASSED (Exit 0)")
        except Exception as e:
            print(f"Error running validator: {e}")
            return
            
    # 2. Package the Zip File
    print("\n[2/2] Zipping Submission Package...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"TeamName_submission_{timestamp}.zip"
    
    # Files/folders to include as per the rules
    folders_to_zip = [
        'output/matching_results.tsv',
        'output/candidate_pairs.tsv',
        'code/business_entity_resolution',
        'Documentation_template.md'
    ]
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in folders_to_zip:
            if not os.path.exists(folder):
                print(f"Missing required folder/file: {folder}")
                continue
                
            if os.path.isfile(folder):
                zipf.write(folder, folder)
            else:
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        # Exclude cache files and data (rules state to only zip code/output/docs)
                        if '__pycache__' not in root and not file.endswith('.pyc'):
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, file_path)
                            
    print(f"\nSuccess! Ready for upload: {zip_filename}")
    print("Upload 'output/matching_results.tsv' to the portal and keep the zip for final review.")

if __name__ == "__main__":
    package_submission()
