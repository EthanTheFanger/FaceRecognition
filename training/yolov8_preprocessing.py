import argparse
import shutil
import yaml
import os

from PIL import Image
from pathlib import Path

# directory variables
wf_dir = "../data/wider_faces_dataset"
yolo_dir = "../data/yolov8_dataset"


def yolo_dirs(target="../data/", remove=False):
    """
    removes and then creates yolo_dirs and file structure
    
    Args:
        target (str): the target directory
        remove (bool): whether to remove the existing directory
    
    returns: root, labels_path, imgs_path
    """
    root = os.path.join(target, "yolov8_dataset")
    if os.path.exists(root) and remove:
        shutil.rmtree(root)
        print("directory removed")
        return
    
    if os.path.exists(root):
        print(f"Directory {root} already exists")
    else:
        os.makedirs(root)

    labels_path = os.path.join(root, "labels")
    imgs_path = os.path.join(root, "images")

    for path in [labels_path, imgs_path]:
        for split in ["train", "val"]:
            split_path = os.path.join(path, split)
            if os.path.exists(split_path):
                print(f"Directory {split_path} already exists")
            else:
                os.makedirs(split_path)
    print("directories created")
    return root, labels_path, imgs_path


def create_yaml(path, test_path="", classes=["face"]):
    """
    Creates a yaml file based on the path, assumes that there are the required files (train and val)
    supports optional test images

    Args:
        path (str): the path to the yolov8 dataset
        test_path (str): the path to the testing images
        classes (list): the list of classes that you want
    """
    if os.path.exists(os.path.join(path, "data.yaml")):
        os.remove(os.path.join(path, "data.yaml"))
        print("Removed previous yaml file")
    
    class_dict = {i: classes[i] for i in range(len(classes))}

    data = {
        "path": path,
        "train": "images/train",
        "val": "images/val",
    }
    if test_path:
        data["test"] = test_path
    data["names"] = class_dict
    
    with open(os.path.join(path, "data.yaml"), 'w') as file:
        yaml.dump(data, file, sort_keys=False)
    
    print("yaml file created")


def create_yolo_annotation(chunk, img_width, img_height):
    """
    returns a list of the center_x center_y width height that follows yolo formatting

    Args:
        chunk (list): list of list of integers. It is the x1, y1, w, h parts from the wider faces annotation file
    """
    res = []
    chunk = [i.split()[:4] for i in chunk]
    
    for i in chunk:
        i = [int(j) for j in i]
        x1, y1, w, h = i
        center_x = (x1 + w/2) / img_width
        center_y = (y1 + h/2) / img_height
        width = w / img_width
        height = h / img_height
        res.append(f"0 {center_x} {center_y} {width} {height}\n")
    return res


def populate_yolo(file, img_dir, target_labels, target_imgs):
    counter = 0
    found_counter = 0
    while counter < len(file):
        filename = file[counter]
        found_file = list(Path(img_dir).rglob(filename))
        num_faces = int(file[counter + 1])

        if found_file:
            found_file = found_file[0]
            if num_faces == 0:
                counter += 3
                continue
            
            chunk = file[counter+2:counter+2+num_faces]
            width, height = Image.open(found_file).size

            label_name = Path(filename).stem + ".txt"
            with open(os.path.join(target_labels, label_name), "a") as f:
                for i in create_yolo_annotation(chunk, width, height):
                    f.write(i)
            shutil.copy(found_file, target_imgs)

            found_counter += 1
            if found_counter % 1000 == 0:
                print(f"processed {found_counter} files!")

        counter += 2 + num_faces
    print(f"files processed in total: {found_counter}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="../data/", help="path to create the yolov8 dataset")
    parser.add_argument("-w", "--wf", type=str, default="../data/wider_faces_dataset", help="path to the wider faces dataset")
    args = parser.parse_args()

    wf_dir = args.wf
    yolo_dir = os.path.join(args.path, "yolov8_dataset")

    with open(os.path.join(wf_dir, "wider_face_train_bbx_gt.txt"), "r") as f:
        train_annotations = f.read().splitlines()

    with open(os.path.join(wf_dir, "wider_face_val_bbx_gt.txt"), "r") as f:
        val_annotations = f.read().splitlines()

    yolo_dirs(target=args.path, remove=True)
    yolo_dirs(target=args.path, remove=False)
    create_yaml(yolo_dir)

    print("processing train...")
    populate_yolo(train_annotations, os.path.join(wf_dir, "WIDER_train/images"), os.path.join(yolo_dir, "labels/train"), os.path.join(yolo_dir, "images/train"))
    
    print("processing val...")
    populate_yolo(val_annotations, os.path.join(wf_dir, "WIDER_val/images"), os.path.join(yolo_dir, "labels/val"), os.path.join(yolo_dir, "images/val"))