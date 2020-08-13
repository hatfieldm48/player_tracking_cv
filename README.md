# player_tracking_cv
A repository for working on solutions that combine player tracking data with computer vision methodologies

## File Overview
* wiki_files
  * A set of files I've pulled up to help document this process, which can be seen over on the project's wiki
* README.md
  * This file
* create_play_as_image.py
  * A file to parse the player location json files, and create images of the tracking information, either all of the players, just the ball, color encoded velocity information, or any combination of those.
* fastai_runpass_classification.ipynb
  * A jupyter notebook to use the tracking images, and predict if an play is either a run or a pass, based on the image
* ngs_player_tracking.py
  * A helper python file for working with the json location information
* requirements.txt
  * Required python files
