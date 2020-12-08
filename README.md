# circular.fashion Coding Challenge
### The Task: 
build an Auto Clothing Image Rotation Web App. 
Given an image of a single item of clothing the wrong way up, the web app should return the image with the correct orientation and resizing.

The web app should be built with Django and the Django Rest Framework, ideally using Keras or Tensorflow. 
We like tests and test driven development (TDD), which is something to consider. 
The application only needs to run locally - but if you could deploy it that would be great.


### Example:
If my image is 32 x 28 png and is in the incorrect rotation by 90 degrees anti clockwise, your app should return the image to me, rotated 90 degrees clockwise with the dimensions 28 x 32 png.

### Structure
The code for the Django Web-App can be found in the folder rotationserver. 
The folder nn holds the code for the training of a neural network.
