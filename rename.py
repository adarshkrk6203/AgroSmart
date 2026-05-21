import os

def rename_folders(base_path):
    for folder in os.listdir(base_path):
        old_path = os.path.join(base_path, folder)

        if os.path.isdir(old_path):
            new_name = folder.replace(" - ", "___") \
                             .replace(" ", "_") \
                             .replace("(", "") \
                             .replace(")", "")

            new_path = os.path.join(base_path, new_name)

            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"{folder} → {new_name}")

# Run for both train and test
rename_folders("dataset/train")
rename_folders("dataset/test")

print("✅ Renaming completed!")