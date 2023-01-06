# BocceBallDL

Implementation of a bocce ball referee system based on Computer Vision and Machine Learning using the Python programming language.

## General Info

Game balls and their colors are detected using the YOLO neural network architecture. This was because there were significant lighting changes in the project environment. The lighting changes are from playing the game indoors and outdoors, day and night, and colored lights used in amusement parks. Previously color-based detection methods (in HSV) were implemented, but they failed. Along with the game balls, the lines indicating the game field are also detected to check for fouls and balls out of bounds.

There are two separate game fields, each measuring 12 x 3 meters. The field is divided into four equal zones of 3 x 3 meters. Each zone is covered by a camera with a wide lens installed three meters above the zone's center.

Balls and lines are constantly monitored, and balls in different parts of the court are taken into account. After the detection phase, the distance between the colored and white balls is calculated, and the balls are ranked based on this criteria.

There were a number of lighting conditions and ball positioning inside the field that were tested as part of this project.

## Steps of the Project

1. Choosing camera models based on the required range they must cover on the game field

2. Putting up scaffolding and installing the cameras

3. Specifying the game field and its four zones by putting black ducktapes on the ground

4. Implementing distortion correction and exact point-to-point mapping

	- Using a printed chess field with the size of 3 x 3 meters

	- Methods from [OpenCV](https://opencv.org)

	- Two different chess fields were used. One with 5 centimeters tiles and one with 10 centimeters tiles.

	- Each point on the camera image is mapped to a position in centimeters inside the chess field. This enables the possibility of calculating the distance in centimeters. An example of one such chess field image is shown below:<br /><br /> <img alt="Chess field image" width="300" src="/chessBoard5.jpg"/>

5. Gathering images of balls in various situations

	- Minimum of 1000 images per color ball

	- Various lighting conditions

	- Different positioning around the zone, with balls covering each other from the camera's sight

6. Labeling color and white balls and zone lines

	- Using [RoboFlow](https://roboflow.com)

7. Applying image augmentation

	- Increasing the dataset by a factor of around six times

9. Training the YOLOv5s network

10. Implementing frame fetching and detection

	- Frame fetching and detection are done in two separate threads

	- Each frame is fetched and pre-processed and then put in a shared variable between two threads

	- The output of detection is a list of balls and their center position

11. Calibrate the positions using trigonometry

	- Because the camera is at the center of the zone, balls that are further from the center will have an offset because of the camera's angle

	- Using trigonometry, the center of the ball is calculated as if it is above the ground by the distance of the colored ball's radius.

12. Checking for fouls

	- Fouls consist of the player's foot breaking the lines around the game field and balls that are out of the game field

13. Determining the distance between colored and white balls and rank the colored balls based on this distance
