# FaceRecognition (in progress)

This project deals with both facial detection and facial recognition.

**Facial detection** answers the question of whether or not there is a face in a photo. A custom YOLOv8 model is used to draw bounding boxes around detected faces. Optionally, detected faces can be passed into DeepFace for age, ethnicity, and emotion estimation. If no faces are detected, the user is notified.

**Facial recognition** takes an anchor image (the face to check for) and a positive image (the image to check against), and determines whether the two images contain the same person. A siamese network trained on the LFW dataset is used for this task.

Facial detection is also used as a preprocessing step for facial recognition — cropping faces from images before passing them into the siamese network makes the inputs cleaner and improves accuracy.

---

## Datasets

- [Labeled Faces in the Wild (LFW)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
- [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)

---

## File Structure

```
FaceRecognition/
|-- training/               # model training scripts
|   |-- face_detection/
|   |-- face_recognition/
|-- inference/              # running trained models
|-- api/                    # FastAPI backend
|-- frontend/               # simple frontend
|-- data/                   # excluded from repo (see setup below)
|   |-- lfw_dataset/
|   |-- wider_faces_dataset/
|-- notebooks/              # notebooks used for experimentation
|-- docker/                 # Dockerfile and config
|-- requirements.txt
|-- README.md
```

---

## Setup

Download the datasets and place them in the `data/` directory following the structure above.

- WIDER FACE: download `WIDER_train`, `WIDER_val`, `wider_face_train_bbx_gt.txt`, and `wider_face_val_bbx_gt.txt`
- LFW: download the dataset from Kaggle linked above

---

## Usage

### Preprocessing

Convert the WIDER FACE dataset into YOLOv8 format before training the face detection model.

```bash
python3 training/face_detection/preprocessing.py -p <path_to_data> -w <path_to_wider_faces>
```

**Arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `-p` | Directory to create the YOLOv8 dataset in | `../data/` |
| `-w` | Path to the WIDER FACE dataset | `../data/wider_faces_dataset` |

**Example:**

```bash
python3 training/face_detection/preprocessing.py -p ../data/ -w ../data/wider_faces_dataset
```

This will create a `yolov8_dataset/` folder at the target path with the following structure:

```
yolov8_dataset/
|-- data.yaml
|-- images/
|   |-- train/
|   |-- val/
|-- labels/
|   |-- train/
|   |-- val/
```
