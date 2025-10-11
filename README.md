# Moving particles counter
This is a `python` (>=3.2) script to count moving particles in a **stationary** video.
It is based on:
  - https://www.educative.io/answers/background-subtraction-opencv
  - https://www.geeksforgeeks.org/find-the-solidity-and-equivalent-diameter-of-an-image-object-using-opencv-python/
  - https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
  - https://www.geeksforgeeks.org/python-opencv-background-subtraction

## Includes
  - `moving_particles_counter.py` main an **only** script
  - `tesst_video.mp4` test video of *Artemia salina* kept in culture.

## `python` dependencies
  - `cv2` https://pypi.org/project/opencv-python/
  - `numpy` https://numpy.org/
  - `pandas` https://pandas.pydata.org/ 
  - `argparse` https://docs.python.org/3/library/argparse.html

## Usage and test
The script is run using `python moving_particles_counter.py -i VIDEO_FILE_NAME -x 1`.
The system can be tested using `python moving_particles_counter.py -i test_video.mp4 -x 1`.

## License
 CC0-1.0
