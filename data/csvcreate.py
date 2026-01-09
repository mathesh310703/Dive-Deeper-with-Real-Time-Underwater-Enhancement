import os
import csv

# Updated dataset folder (no spaces)
dataset_root = r"C:\Users\mathe\OneDrive\Desktop\visionenhancementinwarscenarios\data"

distorted_dir = os.path.join(dataset_root, "distorted")
restored_dir = os.path.join(dataset_root, "restored")

# Check if folders exist
if not os.path.exists(distorted_dir):
    raise FileNotFoundError(f"{distorted_dir} does not exist!")
if not os.path.exists(restored_dir):
    raise FileNotFoundError(f"{restored_dir} does not exist!")

# Get sorted lists of images
distorted_files = sorted(os.listdir(distorted_dir))
restored_files = sorted(os.listdir(restored_dir))

assert len(distorted_files) == len(restored_files), "Mismatch between distorted and restored images!"

# Prepare CSV rows
dataset_rows = []
for d, r in zip(distorted_files, restored_files):
    dataset_rows.append([f"distorted/{d}", f"restored/{r}"])

# Write CSV
csv_path = os.path.join(dataset_root, "dataset.csv")
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["distorted", "restored"])  # header
    writer.writerows(dataset_rows)

print(f"CSV file created at {csv_path} with {len(dataset_rows)} entries!")
