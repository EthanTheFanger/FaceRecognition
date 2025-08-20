# FaceRecognition (in progress)

This project deals with both facial detection and facial recognition. 

Facial detection deals with answering the question of whether or not there is a face in a photo, and facial recognition deals with taking an example photo and then comparing it to an input photo and determining whether or not the example photo is the sae as the input photo. 

For facial detection, I am going to be building my own YOLOv8-face model, whose job it is to bound boxes around faces. Then if the user requests the estimation of their age, ethnicity or emotion, my model will feed that into Deepface's facial analysis. Upon uploading a picture, a cropped version of a face on that picture will be outputted, and the estimation of their age, ethnicity and emotion, if desired. Of course, there will be a notice is there are no faces detected in the picture.

Facial detection, will also be used as a preprocessing step to facial recognition to make the program more rebust, as having cropped faces makes the images much cleaner and accurate for the facial recognition software. 

For facial recognition, I am going to be buiding a simple siamese network, which will have users include an anchor image. An anchor image is the face that we want to check for. Users will also be required to include a positive image, which is the image where we want to check if the anchor matches. The model will be trained on the lfw (labelled faces in the wild) dataset.

This project will be mainly split into two main poritons, training and then deploying. Training deals with building the models and dealing with facial detection and recognition. Deploying will deal with the extractig, transforming and loading of the images as well as API documenting and creating. As well as making a simple front end framework.

## Dataset

This project uses the following datasets, all from kaggle:
- [Labeled Faces in the Wild (LFW)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
- [WIDER FACE](https://www.kaggle.com/datasets/aiacademymaterials/wider-face-detection)

## file structure

facial_project/ <br>
|-- training/ # all model training scripts <br>
| |-- face_detection/ <br>
| |-- face_recognition/ <br>
|-- inference/ # running trained models <br>
|-- api/ # fastAPI backend <br> 
|-- frontend/ # simple frontend <br>
|-- data/ # this is not actually in the repository, and is excluded to keep the repo lightweight <br>
| |-- lfw_dataset/ <br>
| |-- WIDER_face/ <br>
|-- notebooks/ # notebooks that were used for experimenting with different models <br>
|-- docker/ # docker file and config <br>
|-- requirements.txt # dependencies <br>
|-- README.md # project overview :) <br>
