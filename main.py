import cv2
import pickle
import cvzone
import numpy as np

# Video feed
cap = cv2.VideoCapture('carPark.mp4')

width, height = 107, 48  # measures of parking spaces

with open('CarParkPos', 'rb') as f: # import parking space map
    posList = pickle.load(f)

def checkParkingSpace(imgPro):
    spaceCounter = 0

    for pos in posList:
        '''
        crop frames to parking spaces and evaluate their pixels to identify free and occupied parking spaces
        '''
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        # cv2.imshow(str(x * y), imgCrop)
        count = cv2.countNonZero(imgCrop)      #count pixels

        if count < 900:             #free space
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:                       #occupied space
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,thickness=2, offset=0, colorR=color)   #display pixel count

    cvzone.putTextRect(img, f'Free Spaces: {spaceCounter}/{len(posList)}', (350, 50), scale=3,thickness=3, offset=20, colorR=(0, 200, 0))

def imgprocess(img):
    '''
    process the original video frames to identify free and occupied parking spaces
    '''
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDone = cv2.dilate(imgMedian, kernel, iterations=1)  # noise removal

    return imgDone


while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):  # current frame = total frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)                              # reset video to initial frame - loop the video

    success, img = cap.read()

    imgDone = imgprocess(img)

    checkParkingSpace(imgDone)

    #cv2.imshow("Image2", imgDone)
    cv2.imshow("Image", img)

    cv2.waitKey(10)