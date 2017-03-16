# Face The Gate

Recognize faces using `OpenCV` and `Python`.


### Getting Things Ready
* Install `OpenCV` with `Python` bindings.
* Clone and get started - 
  ```sh
  $ git clone git@github.com:singhpratyush/face-the-gate.git
  $ cd face-the-gate/
  $ mkdir rsc/images
  ```

### Adding New Face Data
* While in `src`, use `add_data.py` to add new face data.
* Arguments -
  * `-c` | `--camera-id`  - Camera device ID. Defaults to 0.
  * `-i` | `--subject-id` - ID of subject whise data is to be added.
  * `-s` | `--start-pos`  - Position of start index for the subject ID.
  * `-e` | `--end-pos`    - Position of end index for the subjet ID.

### Testing
* While in `src`, use `main.py` to test the data collected. You must have atleast 2 subjects registered to start this activity.
* Make sure to use `-r` or `--refresh-data` option to rebuild the classification data from the images.
* You may use the `-c` or `--camera-id` to specify the camera ID if default is not 0.

### Classifiers Used
Uses 3 different classifiers - 
* Eigen Face Recognizer
* Fisher Face Recognizer
* LBPH Face Recognizer

### Rotation to Straighten Images

Sometimes, camera is little bent and the resulting image of face is little bent too.
But this small tilt has a large impact on the recognition numbers. To overcome this, following technique is used -
* Get the position of eyes in the frame.
* Calculate the angle by which they are away from being on the same horizontal line.
  ```python
    delta_y = right_eye_y_center - left_eye_y_center
    delta_x = right_eye_x_center - left_eye_x_center
    rotation_degrees = math.degrees(math.atan(float(delta_y) / (delta_x)))
  ```
* Generate a 2D rotation matrix for the corresponding canvas and rotation angle.
* Perform affine transform using the rotation matrix.

