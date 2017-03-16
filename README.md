# Face The Gate

Recognize faces using `OpenCV` and `Python`.

### Classifiers
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
