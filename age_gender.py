import cv2 as cv
import time
import argparse
import os

# Constants - Verify these paths are correct
FACE_PROTO = "./models/opencv_face_detector.pbtxt"
FACE_MODEL = "./models/opencv_face_detector_uint8.pb"
AGE_PROTO = "./models/age_deploy.prototxt"
AGE_MODEL = "./models/age_net.caffemodel"
GENDER_PROTO = "./models/gender_deploy.prototxt"
GENDER_MODEL = "./models/gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGELIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDERLIST = ['Male', 'Female']

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Age and gender recognition using OpenCV.')
    parser.add_argument('--input', help='Path to input image or video file. Skip for camera.')
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="Device to use for inference")
    args = parser.parse_args()

    # Verify model files exist
    for file_path in [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Load Networks with proper methods
    face_net = cv.dnn.readNet(FACE_MODEL, FACE_PROTO)  # For .pb and .pbtxt
    age_net = cv.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)  # For Caffe models
    gender_net = cv.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)  # For Caffe models

    # Device Handling
    if args.device == "cpu":
        face_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        face_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    elif args.device == "gpu":
        face_net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        face_net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    # Open video source
    cap = cv.VideoCapture(args.input if args.input else 0)
    padding = 20
    frame_count = 0
    start_time = time.time()

    while True:
        # Read frame
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        frame_count += 1
        frameFace, bboxes = getFaceBox(face_net, frame)
        
        if not bboxes:
            print("No face detected, checking next frame")
            continue

        for bbox in bboxes:
            # Extract face with padding
            face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                        max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
            
            if face.size == 0:
                continue

            # Resize and preprocess face
            face = cv.resize(face, (227, 227))
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Gender Prediction
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDERLIST[gender_preds[0].argmax()]

            # Age Prediction
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGELIST[age_preds[0].argmax()]

            # Display results
            label = f"{gender}, {age}"
            cv.putText(frameFace, label, (bbox[0], bbox[1]-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)

        # Calculate and display FPS
        fps = frame_count / (time.time() - start_time)
        cv.putText(frameFace, f"FPS: {fps:.2f}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv.imshow("Age Gender Detection", frameFace)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), 
                         (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

if __name__ == "__main__":
    main()