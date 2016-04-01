import math
import sys
from getopt import getopt, GetoptError

import cv2

# Path to cascades
FRONTAL_FACE_CASCADE_PATH = "../rsc/cascades/haarcascade_frontalface_default.xml"
RIGHT_EYE_CASCADE_PATH = "../rsc/cascades/haarcascade_righteye_2splits.xml"
LEFT_EYE_CASCADE_PATH = "../rsc/cascades/haarcascade_lefteye_2splits.xml"


def get_new_data(camera_id, start_pos = 0, end_pos = 10, subject_id = 0):

    # Open Video Capture
    cap = cv2.VideoCapture(camera_id)

    # Check if camera is working properly
    if not cap.isOpened():
        return

    # Make camera ready
    for _ in range(0, 50):
        cap.read()

    # Initialize cascades
    FC = cv2.CascadeClassifier(FRONTAL_FACE_CASCADE_PATH)
    REC = cv2.CascadeClassifier(RIGHT_EYE_CASCADE_PATH)
    LEC = cv2.CascadeClassifier(LEFT_EYE_CASCADE_PATH)

    count = start_pos
    streak = 0

    # Start the recognition process
    while True:
        ret, frame = cap.read()

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        normal = cv2.equalizeHist(grey)

        faces = FC.detectMultiScale(
            normal,
            scaleFactor=2,
            minNeighbors=2,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        if len(faces) == 1:
            face = faces[0]

            (x, y, w, h) = face

            close_up_face = cv2.equalizeHist(grey[y:y + h, x:x + h])
            left_eyes = LEC.detectMultiScale(
                close_up_face,
                scaleFactor=1.6,
                minNeighbors=4,
                minSize=(5, 5)
            )
            right_eyes = REC.detectMultiScale(
                close_up_face,
                scaleFactor=1.6,
                minNeighbors=4,
                minSize=(5, 5)
            )

            if len(left_eyes) > 0 and len(right_eyes) > 0:

                # Identify right and left eyes
                left_eye_x_center = left_eyes[0][0] + left_eyes[0][2] / 2
                left_eye_y_center = left_eyes[0][1] + left_eyes[0][3] / 2

                right_eye_x_center = right_eyes[0][0] + right_eyes[0][2] / 2
                right_eye_y_center = right_eyes[0][1] + right_eyes[0][3] / 2

                for (m, n, o, p) in left_eyes[1:]:
                    if m + o / 2 > left_eye_x_center:
                        left_eye_x_center = m + o / 2
                        left_eye_y_center = n + p / 2

                for (m, n, o, p) in right_eyes[1:]:
                    if m + o / 2 < right_eye_x_center:
                        right_eye_x_center = m + o / 2
                        right_eye_y_center = n + p / 2

                if left_eye_x_center != right_eye_x_center:

                    # Draw rectangles and circles in the original frame
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255)
                    )

                    cv2.circle(
                        frame,
                        (x + left_eye_x_center, y + left_eye_y_center),
                        1,
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(
                        frame,
                        (right_eye_x_center, right_eye_y_center),
                        1,
                        (0, 255, 0),
                        2
                    )

                    # Get rotation degree
                    rotation_degree = math.degrees(
                        math.atan(
                            float(right_eye_y_center - left_eye_y_center) / (right_eye_x_center - left_eye_x_center)
                        )
                    )

                    # Rotate the canvas and perform further operations
                    if math.fabs(rotation_degree) < 15:
                        w_m, h_m = normal.shape
                        rotation_matrix = cv2.getRotationMatrix2D((h_m / 2, w_m / 2), rotation_degree, 1.0)
                        normal = cv2.warpAffine(normal, rotation_matrix, (h_m, w_m))

                        new_x, new_y, new_w, new_h = new_face = FC.detectMultiScale(
                            normal,
                            scaleFactor=2,
                            minNeighbors=2,
                            minSize=(30, 30),
                            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                        )[0]

                        new_face_closeup = normal[new_y:new_y + new_h, new_x:new_x + new_w]
                        streak += 1
                        if streak == 5:
                            img_src = '../rsc/images/{0}_{1}.jpg'.format(subject_id, count)
                            cv2.imwrite(img_src, new_face_closeup)
                            print('Saved ' + img_src + ' Size : ' + str(new_face_closeup.shape))
                            count += 1
                            streak = 0
                        cv2.waitKey(100)
                    else:
                        streak = 0
                else:
                    streak = 0
            else:
                streak = 0
        else:
            streak = 0

        cv2.imshow('Image', frame)

        if count > end_pos:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            print('Quitting')
            break


if __name__ == "__main__":

    # Constants
    camera_id = 0
    has_start = False
    has_end = False
    start_pos = 0
    end_pos = 10
    subject_id = -1

    options = ()
    # Process options
    try:
        options, args = getopt(
            sys.argv[1:],
            "c:s:e:i:",
            [
                '--camera_id=',
                '--start-pos=',
                '--end-pos=',
                '--subject-id='
            ]
        )
    except GetoptError as err:
        print('Invalid arguments')

    for opt, arg in options:
        if opt in ('-c', '--camera-id'):
            camera_id = int(arg)
        if opt in ('-s', '--start-pos'):
            start_pos = int(arg)
            has_start = True
        if opt in ('-e', '--end-pos'):
            end_pos = int(arg)
            has_end = True
        if opt in ('-i', '--subject-id'):
            subject_id = int(arg)

    if has_start and not has_end:
        print("Not provided end position.")
        end_pos = start_pos + 10
    if has_end and not has_start:
        print('Not provided start position')
        if end_pos > 10:
            start_pos = end_pos - 10
        else:
            start_pos = 0
    if has_end and has_start and (start_pos > end_pos):
        print('Provided both in reverse order')
        start_pos += end_pos
        end_pos = start_pos - end_pos
        start_pos -= end_pos

    get_new_data(camera_id=camera_id, start_pos=start_pos, end_pos=end_pos, subject_id=subject_id)