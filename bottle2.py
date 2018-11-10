import module_manager
module_manager.ignore_module('imutils.video')
module_manager.review()
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import pytesseract


class meds():
    def __init__(self, name, period, count):
        self.name = name
        self.period = period
        self.count = count

    def takePillNow(self):
        x =  time.localtime()
        if x[3] == self.period:
            return True
        if x[3] == self.period - 1:
            if x[4] > 30:
                return True
        return False

    def takePillString(self):
        return "Take $d pills" %d(self.count)

r815137 = "815137"
r844275 = "844275"
ibu = "Ibuprofe"
colbottle = (255,255,255)

bottlecoords = []
bottleLst = []
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

IGNORE = set(["background", "aeroplane", "bicycle", "bird", "boat", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"])
# COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print(CLASSES, IGNORE)
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
		# the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            # print(CLASSES)
            if CLASSES[idx] in IGNORE: continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            bottleLst.append(startX)
            bottleLst.append(startY)
            bottleLst.append(endX)
            bottleLst.append(endY)


            bottlecoords.append(bottleLst)
            bottleLst = []
            #print('bottlecoord',bottlecoords)

            camera = cv2.VideoCapture(0)
            i = 1
            return_value, image = camera.read()
            cv2.imwrite('opencv' + str(i) + '.png', image)
            img = cv2.imread('/Users/kellyyu/PycharmProjects/new/venv/bin/opencv' + str(i) + '.png', 0)
            #crop_img = img[bottlecoords[-1][1]:bottlecoords[-1][3], bottlecoords[-1][0]:bottlecoords[-1][2]]
            #if bottlecoords[-1][1] > 0 and bottlecoords[-1][0] > 0:
            #    print(bottlecoords[-1][1], bottlecoords[-1][3], bottlecoords[-1][0], bottlecoords[-1][2])
            text = pytesseract.image_to_string(img)
            print(text)
            if "815137" in text or 'a PALE YELLOW' in text or 'BLET imprinted with UO' in text or '03297309' in text or 'PALE' in text:
                label = "Take 1 tab by mouth"
            else:
                label = "Don't take this now"

            del camera



            # draw the prediction on the frame

            # cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colbottle, 2)


	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()


# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
