import json
import os

INPUT_FILE = "/mnt/c/Users/Jorge/Desktop/FINALBOSSTest/result_coco.json"       # your COCO-style file
OUTPUT_FILE = "/mnt/c/Users/Jorge/Desktop/FINALBOSSTest/annotations_labelstudio.json"

# Folder inside your LS Local Storage (modify to match your setup)
LS_FOLDER = "/mnt/c/Users/Jorge/Desktop/FINALBOSSTest/datasets/labelstudio_storage/chosen_images"


def coco_polygon_to_ls(points, img_width, img_height):
    """
    Convert COCO flat polygon into Label Studio normalized polygon points.
    """
    ls_points = []
    for i in range(0, len(points), 2):
        x, y = points[i], points[i+1]
        x_pct = (x / img_width) * 100
        y_pct = (y / img_height) * 100
        ls_points.append([x_pct, y_pct])
    return ls_points


def main():
    with open(INPUT_FILE, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    # Build a map: image_id -> list of annotations
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    ls_tasks = []

    for image_id, anns in anns_by_image.items():

        img = images[image_id]
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]

        image_url = f"/data/local-files/?d={LS_FOLDER}/{filename}"

        results = []

        for ann in anns:

            # Skip empty segmentation
            if not ann.get("segmentation"):
                print(f"⚠️ Skipping empty segmentation for ann {ann['id']}")
                continue

            # Some COCO export tools store segmentation like [[]]
            seg_raw = ann["segmentation"][0]
            if len(seg_raw) < 6:
                print(f"⚠️ Skipping invalid polygon (too few points) for ann {ann['id']}")
                continue

            ls_polygon = coco_polygon_to_ls(seg_raw, img_width, img_height)

            results.append({
                "id": f"poly-{ann['id']}",
                "type": "polygonlabels",
                "value": {
                    "points": ls_polygon,
                    "polygonlabels": ["bubble"]
                },
                "from_name": "label",
                "to_name": "image",
                "origin": "manual"
            })

        # Build final LS task
        task = {
            "data": {
                "image": image_url
            },
            "annotations": [
                {"result": results}
            ]
        }

        ls_tasks.append(task)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(ls_tasks, f, indent=2)

    print(f"🎉 Done! Saved Label Studio tasks to: {OUTPUT_FILE}")
    print(f"Total tasks: {len(ls_tasks)}")


if __name__ == "__main__":
    main()