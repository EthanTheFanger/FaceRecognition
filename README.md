# FaceRecognition

This project deals with both facial detection and facial recognition. 

Facial detection deals with answering the question of whether or not there is a face in a photo, and facial recognition deals with taking an example photo and then comparing it to an input photo and determining whether or not the example photo is the sae as the input photo. 

Facial detection, in this project, is used as a preprocessing step to facial recognition to make the program more rebust, as having cropped faces makes the images much cleaner and accurate for the facial recognition software. 

For facial detection, I am going to be using deepface to crop the faces. On top of that, users are also able to request the estimation of their age, ethnicity and emotion as a fun way of using facial detection. 

For facial recognition, I am going to be using a simple siamese network, which will have users include an anchor image, which is the face that we want to check for, and a positive image, which is the image where we want to check if the anchor matches. 

I will also be deploying an API. 