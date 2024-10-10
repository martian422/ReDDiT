import os
import zipfile

def collect_files(root_folder, exclude_list):
    collected_files = []
    for root, dirs, files in os.walk(root_folder):
        # Exclude soft linked files and folders
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
        files = [f for f in files if not os.path.islink(os.path.join(root, f))]
        
        # Exclude folders with names in exclude_list
        dirs[:] = [d for d in dirs if d not in exclude_list]
        
        # Add files to collected_files
        collected_files.extend([os.path.join(root, file) for file in files])
    return collected_files

def zip_files(file_list, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in file_list:
            zipf.write(file, os.path.relpath(file))

def main():
    folder_to_backup = '/home/MaTianren/Workspace/MDLM-neo'
    exclude_list = ['wandb', 'gsm8k', 'outputs','wandb', 'var', '__pycache__'] # List of folders to exclude
    backup_zip_filename = 'MDM-1009.zip'

    collected_files = collect_files(folder_to_backup, exclude_list)
    zip_files(collected_files, backup_zip_filename)

    print("Backup completed successfully!")

if __name__ == "__main__":
    main()
