import cv2
import math
import time
import dlib


carCascade = cv2.CascadeClassifier('vech.xml')
video = cv2.VideoCapture('video.mp4')

WIDTH = 720
HEIGHT = 720

def estimateSpeed(p1, p2):    
    d_pixels = math.sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))
    ppm = 8.0       # ppm = p2[2] / carWidth
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed

def trackMultipleObjects():
    rectangleColor = (0, 255, 0)    # green color
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carDabba = {}
    carLoc1 = {}
    carLoc2 = {}
    speed = [None] * 1000

    out = cv2.VideoWriter('outNew.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH, HEIGHT))

    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImg = image.copy()

        frameCounter = frameCounter + 1
        carIDtoDelete = []

        for carID in carDabba.keys():
            dabbaQuality = carDabba[carID].update(image)
            
            if dabbaQuality < 7:
                carIDtoDelete.append(carID)

        #removing cars from the list of trackers and their locations too
        for carID in carIDtoDelete:
            carDabba.pop(carID, None)
            carLoc1.pop(carID, None)
            carLoc2.pop(carID, None)
        
        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      #converting to gray salce
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24)) 

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carDabba.keys():
                    trackPos = carDabba[carID].get_position()

                    t_x = int(trackPos.left())
                    t_y = int(trackPos.top())
                    t_w = int(trackPos.width())
                    t_h = int(trackPos.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                # adding new tracker                    
                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    carDabba[currentCarID] = tracker
                    carLoc1[currentCarID] = [x, y, w, h]
                    currentCarID = currentCarID + 1

        for carID in carDabba.keys():
            trackPos = carDabba[carID].get_position()
            t_x = int(trackPos.left())
            t_y = int(trackPos.top())
            t_w = int(trackPos.width())
            t_h = int(trackPos.height())
            
            cv2.rectangle(resultImg, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
            carLoc2[carID] = [t_x, t_y, t_w, t_h]

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0/(end_time - start_time)

        for i in carLoc1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLoc1[i]
                [x2, y2, w2, h2] = carLoc2[i]

                carLoc1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x1, y2, w2, h2])

                    if speed[i] != None and y1 >= 180:
                        cv2.putText(resultImg, str(int(speed[i])) + "km/h", (int(x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 100) ,2)

        cv2.imshow('result', resultImg)

        out.write(resultImg)

        if cv2.waitKey(1) == 27:
            break

    
    cv2.destroyAllWindows()
    out.release()

if __name__ == '__main__':
    trackMultipleObjects()
