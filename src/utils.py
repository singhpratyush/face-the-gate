import math
import os

import cv2

import src.constants


def refresh_data():

    # Get path to all images
    path = 'rsc/images'
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    # Initialize empty lists for these images and labels
    images = []
    labels = []

    for path in image_paths:
        images.append(cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE))
        labels.append(int(os.path.split(path)[1].split("_")[0]))

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
    eigen_recognizer.load('rsc/recognition_files/eigen_recognizer.yml')
    fisher_recognizer.load('rsc/recognition_files/fisher_recognizer.yml')
    lbhp_recognizer.load('rsc/recognition_files/lbhp_recognizer.yml')

    # Initialize cascades
    FC = cv2.CascadeClassifier("rsc/cascades/haarcascade_frontalface_default.xml")
    REC = cv2.CascadeClassifier("rsc/cascades/haarcascade_righteye_2splits.xml")
    LEC = cv2.CascadeClassifier("rsc/cascades/haarcascade_lefteye_2splits.xml")

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

            close_up_face = face[y:y + h, x:x + h]
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
                        (left_eye_x_center, left_eye_y_center),
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
                            float(right_eye_y_center - left_eye_y_center) / (right_eye_x_center - right_eye_y_center)
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

                        eigen_prediction = eigen_recognizer.predict(new_face_closeup)
                        fisher_prediction = fisher_recognizer.predict(new_face_closeup)
                        lbhp_prediction = lbhp_recognizer.predict(new_face_closeup)

                        # Do something with these predictions
                        print(
                            "\nEigen Prediction - ", eigen_prediction,
                            "\nFisher Prediction - ", fisher_prediction,
                            "\nLBHP Prediction - ", lbhp_prediction
                        )
        cv2.imshow('Image', frame)

        # Terminate loop if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Quitting')
            break