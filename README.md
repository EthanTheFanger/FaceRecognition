# FaceRecognition (in progress)

This project deals with both facial detection and facial recognition. 

Facial detection deals with answering the question of whether or not there is a face in a photo, and facial recognition deals with taking an example photo and then comparing it to an input photo and determining whether or not the example photo is the sae as the input photo. 

Facial detection, in this project, is used as a preprocessing step to facial recognition to make the program more rebust, as having cropped faces makes the images much cleaner and accurate for the facial recognition software. 

For facial detection, I am going to be using deepface to detect faces. Users will be able to use facial detection seperately from facial verification. Upon uploading a picture, a cropped version of a face on that picture will be outputted. Of course, there will be a notice is there are no faces detected in the picture. On top of that, users are also able to request the estimation of their age, ethnicity and emotion as a fun way of using facial detection. 

For facial recognition, I am going to be buiding a simple siamese network, which will have users include an anchor image. An anchor image is the face that we want to check for. Users will also be required to include a positive image, which is the image where we want to check if the anchor matches. The model will be trained on the lfw (labelled faces in the wild) dataset.

This project will be mainly split into two main poritons, training and then deploying. Training deals with building the models and dealing with facial detection and recognition. Deploying will deal with the extractig, transforming and loading of the images as well as API documenting and creating. As well as making a simple front end framework.
