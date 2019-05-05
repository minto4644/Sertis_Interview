# Sertis Assignment for Machine Learning Engineer Role

## Problem statement:
Imagine we have a model deployed on the cloud which performs face recognition on images sent to it. This model works great on well-oriented images, i.e. images which are the right way up. However, when badly-oriented images are sent, e.g. upside-down images, the model performs poorly. Since we have no  control over how the images are sent and have no guarantee that the images will come with orientation-metadata, we would like a pre-processing step which fixes the orientation of the images before being sent to the main model. This is where you come in.

          


## Please attempt the following:

                   

- Design a solution which classifies the orientation of any image of a face into one of four categories. The four orientations considered are shown in the image. 

- Create an API which takes in images of faces and returns them with their orientation fixed. This could be a RESTful API or just a class which contains your model.

- [Optional] Write some unit tests for your API.

- [Optional] Deploy your API somewhere, e.g., heroku.
