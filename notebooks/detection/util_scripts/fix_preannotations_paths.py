import json
import os

INPUT_FILE = "/mnt/c/Users/Jorge/Desktop/FINALBOSSTest/maskrcnn_bubble_annotations.json"          # your input file
OUTPUT_FILE = "/mnt/c/Users/Jorge/Desktop/FINALBOSSTest/preannotations_fixed.json"       # output for LS import

# Change this to match your LS Local Storage folder name
LS_FOLDER = "/mnt/c/Users/Jorge/Desktop/FINALBOSSTest/datasets/labelstudio_storage/chosen_images"

def fix_path(old_path):
    """
    Extracts the filename from the old absolute Windows path
    and converts it to a Label Studio local-files URL.
    """
    # Extract the "d=" query part
    if "?d=" in old_path:
        old_path = old_path.split("?d=")[1]

    filename = os.path.basename(old_path)
    return f"/data/local-files/?d={LS_FOLDER}/{filename}"

def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    for item in data:
        old_path = item["data"]["image"]
        new_path = fix_path(old_path)
        item["data"]["image"] = new_path

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Done! Fixed file saved at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
