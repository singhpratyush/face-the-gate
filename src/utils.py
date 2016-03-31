import cv2


def refresh_data():
    return


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

