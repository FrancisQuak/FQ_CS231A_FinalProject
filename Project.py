# Import library
import numpy as np
import cv2

# Define background subtractor
backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold = 400, detectShadows=True)
# detectShadows will paint shadows in a different color
backSub.setShadowThreshold(0.9)
backSub.setkNNSamples(1)

# Initialise video stream
cap = cv2.VideoCapture('TrafficVid.mp4')
if not cap.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
    
# Class to store tracked vehicles
class Vehicle: 
    bndBox = None
    count=0
    max_life = 10
    life = 10

    corrected = True
    num_corrected = 0
    
    def __init__(self, box):
        self.bndBox = box
        self.widths = [box[2]]
        self.heights = [box[3]]

        self._kalman_initial_pos = None
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
            ], np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
            ], np.float32)
        self.kalman.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 10]
            ], np.float32) * 0.50
        self.kalman.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1],
            ], np.float32) * 0.90
        
        pos = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2], np.float32)
        self._kalman_initial_pos = pos
        self.kalman.correct(pos)

    def update(self, box):
        pos = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2], np.float32)
        pos -= self._kalman_initial_pos

        self.kalman.correct(pos) 
        self.corrected = True
        
        self.count += 1
        self.life = self.max_life
        self.bndBox = box

        self.widths.append(box[2])
        self.heights.append(box[3])
        # Averaging width/height for possible smoother boxes drawn
        if len(self.widths) > 1: 
            self.widths.pop(0)
            self.heights.pop(0)

    def predict(self):
        pred = self.kalman.predict()
        
        pred = np.squeeze(pred[:2])
        pred += self._kalman_initial_pos

        self.corrected = False

        self.bndBox = np.array([
            pred[0] - self.bndBox[2] / 2,
            pred[1] - self.bndBox[3] / 2,
            self.bndBox[2],
            self.bndBox[3]
            ], np.float32) 

# Function to determine centre of box
def center_of_box(box): #box => x,y,w,h
    x,y,w,h = box
    return (x+w/2, y+h/2)
# Function to calculate distance between 2 centre points
def dist_btwn_pts(pt1,pt2):
    x1,y1=pt1
    x2,y2=pt2
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

# First loop to learn the mean background representation
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    # Apply background subtractor and set a learn rate
    fgMask = backSub.apply(frame, learningRate=100 / cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fgMask = (fgMask==255)*255

cap = cv2.VideoCapture('TrafficVid.mp4')
# Create a list to store all the vehicle
tracked_veh = []
remove_veh = []
while(True): # 2nd loop
    ret, frame = cap.read()
    if not ret:
        break
    # To resize in order to fit onto monitor
    frame = cv2.resize(frame, (720, 480)) 
    # 2nd round of background subtractor with a lower learn rate
    fgMask = backSub.apply(frame, learningRate=0.01)
    # set all shadow regions to 0
    fgMask[fgMask == backSub.getShadowValue()] = 0
    # Create a 3x3 matrix and apply erosion follow by dilation morphological operations
    kernel = np.ones((3,3),np.uint8)
    fgMask = cv2.erode(fgMask,kernel,iterations=1)
    fgMask = cv2.dilate(fgMask,kernel,iterations=3)
    # To find contours
    contours, hierarchy = cv2.findContours(fgMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # Loop thru to find blinding boxes and draw them
    for cnt in contours:
        box = cv2.boundingRect(cnt)
        area = box[2] * box[3]
        if area < 200: # observed value as threshold
            continue      
        found = False
        for vehicle in tracked_veh:            
            v_pt = center_of_box(vehicle.bndBox)
            d_pt = center_of_box(box)
            if dist_btwn_pts(v_pt,d_pt)<30: # max distance
                found = True
                vehicle.update(box)
                break
        if not found:
            new_v = Vehicle(box)
            tracked_veh.append(new_v)
    # Life cycle for tracked vehicles
    for vehicle in tracked_veh:
        vehicle.life -=1
        if vehicle.life == 0: 
            remove_veh.append(vehicle)
            continue
        x,y,w,h=vehicle.bndBox
        # To draw bounding box
        if vehicle.count >= 5 and vehicle.corrected: # to filter outliers from being drawn
            center_x, center_y = x + w/2, y + h/2
            avg_w = sum(vehicle.widths) / len(vehicle.widths) / 2
            avg_h = sum(vehicle.heights) / len(vehicle.heights) / 2

            _ul = (int(center_x-avg_w), int(center_y-avg_h))
            _lr = (int(center_x+avg_w), int(center_y+avg_h))
            color = (36,255,12) if vehicle.corrected else (10, 10, 255)
            cv2.rectangle(frame, _ul, _lr, color, 2)

        vehicle.predict()

        for v in remove_veh:
            tracked_veh.remove(v)
        remove_veh = []
        
    cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))  
    
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)
    
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break  

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
