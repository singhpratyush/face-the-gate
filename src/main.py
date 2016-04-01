import math
import os
import sys
from getopt import getopt, GetoptError

import cv2
from numpy import array

usage_text = './main.py [-r] [-c <camera_id>]'

# Path to recognizer
EIGEN_RECOGNIZER_PATH = '../rsc/recognition_files/eigen_recognizer.yml'
FISHER_RECOGNIZER_PATH = '../rsc/recognition_files/fisher_recognizer.yml'
LBHP_RECOGNIZER_PATH = '../rsc/recognition_files/lbhp_recognizer.yml'

# Path to cascades
FRONTAL_FACE_CASCADE_PATH = "../rsc/cascades/haarcascade_frontalface_default.xml"
RIGHT_EYE_CASCADE_PATH = "../rsc/cascades/haarcascade_righteye_2splits.xml"
LEFT_EYE_CASCADE_PATH = "../rsc/cascades/haarcascade_lefteye_2splits.xml"


def refresh_data():
    # Get path to all images
    path = '../rsc/images'
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    # Initialize empty lists for these images and labels
    images = []
    labels = []

    for path in image_paths:
        images.append(cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE))
        labels.append(int(os.path.split(path)[1].split("_")[0]))

    labels = array(labels)

    # Create recognizers
    eigen_recognizer = cv2.createEigenFaceRecognizer()
    fisher_recognizer = cv2.createFisherFaceRecognizer()
    lbhp_recognizer = cv2.createLBPHFaceRecognizer()

    # Train recognizers
    eigen_recognizer.train(images, labels)
    fisher_recognizer.train(images, labels)
    lbhp_recognizer.train(images, labels)

    # Save results
    eigen_recognizer.save(EIGEN_RECOGNIZER_PATH)
    fisher_recognizer.save(FISHER_RECOGNIZER_PATH)
    lbhp_recognizer.save(LBHP_RECOGNIZER_PATH)


def start_gate_keeper(camera_id):
    # Open Video Capture
    cap = cv2.VideoCapture(camera_id)

    # Check if camera is woring properly
    if not cap.isOpened():
        return

    # Make camera ready
    for _ in range(0, 50):
        cap.read()

    # Create recognizers
    eigen_recognizer = cv2.createEigenFaceRecognizer()
    fisher_recognizer = cv2.createFisherFaceRecognizer()
    lbhp_recognizer = cv2.createLBPHFaceRecognizer()

    # Load files
    eigen_recognizer.load(EIGEN_RECOGNIZER_PATH)
    fisher_recognizer.load(FISHER_RECOGNIZER_PATH)
    lbhp_recognizer.load(LBHP_RECOGNIZER_PATH)

    # Initialize cascades
    FC = cv2.CascadeClassifier(FRONTAL_FACE_CASCADE_PATH)
    REC = cv2.CascadeClassifier(RIGHT_EYE_CASCADE_PATH)
    LEC = cv2.CascadeClassifier(LEFT_EYE_CASCADE_PATH)

    # For calculating average
    sum_eigen_prediction = 0
    sum_fisher_prediction = 0
    sum_lbhp_prediction = 0
    n_eigen_prediction = 0
    n_fisher_prediction = 0
    n_lbhp_prediction = 0


    # Start recognition task
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

            x, y, w, h = face

            close_up_face = cv2.equalizeHist(grey[y:y + h, x:x + h])
            cv2.imshow('Close Up',  close_up_face)
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
                        (x, x + w),
                        (y, y + h),
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
                        (x + right_eye_x_center, y + right_eye_y_center),
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

                        new_faces = FC.detectMultiScale(
                            normal,
                            scaleFactor=2,
                            minNeighbors=2,
                            minSize=(30, 30),
                            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                        )

                        if len(new_faces) == 0:
                            continue

                        new_x, new_y, new_w, new_h = new_faces[0]
                        new_face_closeup = normal[new_y:new_y + new_h, new_x:new_x + new_w]

                        new_face_closeup = cv2.resize(new_face_closeup, (192,192))

                        eigen_prediction = eigen_recognizer.predict(new_face_closeup)
                        fisher_prediction = fisher_recognizer.predict(new_face_closeup)
                        lbhp_prediction = lbhp_recognizer.predict(new_face_closeup)

                        # Increment averaging parameters
                        sum_eigen_prediction += eigen_prediction[1]
                        sum_fisher_prediction += fisher_prediction[1]
                        sum_lbhp_prediction += lbhp_prediction[1]
                        n_eigen_prediction += 1
                        n_fisher_prediction += 1
                        n_lbhp_prediction += 1

                        # Do something with these predictions
                        #    print('Eigen  Prediction - ', eigen_prediction)
                        #    print('Fisher Prediction - ', fisher_prediction)
                        #    print('LBHP   Prediction - ' , lbhp_prediction)
                        #    print('\n\n')
                        cv2.imshow('Face Closeup', new_face_closeup)

        cv2.imshow('Image', frame)

        # Terminate loop if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            print ('Average Eigen  Prediction = ' + str(sum_eigen_prediction / n_eigen_prediction))
            print ('Average Fisher Prediction = ' + str(sum_fisher_prediction / n_fisher_prediction))
            print ('Average LBHP   Prediction = ' + str(sum_lbhp_prediction / n_lbhp_prediction))
            break


if __name__ == "__main__":

    # Constants
    camera_id = 0

    options = ()
    # Process options
    try:
        options, args = getopt(
            sys.argv[1:],
            "rc:",
            [
                '--refresh-data',
                '--camera_id='
            ]
        )
    except GetoptError as err:
        print('Invalid arguments')

    for opt, arg in options:
        if opt in ('-r', '--refresh-data'):
            refresh_data()
        if opt in ('-c', '--camera-id'):
            camera_id = int(arg)

    start_gate_keeper(camera_id=camera_id)
