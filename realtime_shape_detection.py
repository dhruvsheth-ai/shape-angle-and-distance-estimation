import csv
import sys
import time
import cv2
import numpy as np
from scipy.spatial import distance as dist
from pathlib import Path
import math

def gradient(pt1,pt2):
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
   



def calcDistTri(x3, y3, x2, y2, x, y):

    dist3 = round(dist.euclidean((x3, y3), (x2, y2)))
    dist2 = round(dist.euclidean((x2, y2), (x, y)))
    dist1 = round(dist.euclidean((x, y), (x3, y3)))
    
    #print(dist1)
    cv2.putText(frame, str(dist1), (round(0.5 * x + 0.5 * x3), round(0.5 * y + 0.5 * y3)) , font, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, str(dist2), (round(0.5 * x2 + 0.5 * x), round(0.5 * y2 + 0.5 * y)) , font, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, str(dist3), (round(0.5 * x3 + 0.5 * x2), round(0.5 * y3 + 0.5 * y2)) , font, 0.5, (0, 0, 0), 1)


    pt3 = x3, y3
    pt2 = x2, y2
    pt1 = x, y
    
    m2 = gradient(pt2,pt1)
    n2 = gradient(pt2,pt3)
    if m2 is not None and n2 is not None:
        angR2 = math.atan((n2-m2)/(1+(n2*m2)))
        angD2 = math.degrees(angR2)
        if math.isnan(angD2) is False:
            cv2.putText(frame, str(round(abs(angD2))), (pt2[0]-40,pt2[1]-20), font, 1, (0, 0, 0))        
            #print(round(abs(angD2)),(pt1[0]-40,pt1[1]-20))
    
    m3 = gradient(pt3,pt2)
    n3 = gradient(pt3,pt1)
    if m3 is not None and n3 is not None:
        angR3 = math.atan((n3-m3)/(1+(n3*m3)))
        angD3 = math.degrees(angR3)
        if math.isnan(angD3) is False:
            cv2.putText(frame, str(round(abs(angD3))), (pt3[0]-40,pt3[1]-20), font, 1, (0, 0, 0))        
            #print(round(abs(angD3)),(pt1[0]-40,pt1[1]-20))

    
    m = gradient(pt1,pt3)
    n = gradient(pt1,pt2)
    if m is not None and n is not None:
        angR = math.atan((n-m)/(1+(n*m)))
        angD = math.degrees(angR)
        if math.isnan(angD) is False:
            cv2.putText(frame, str(round(abs(angD))), (pt1[0]-40,pt1[1]-20), font, 1, (0, 0, 0))                
            #print(round(abs(angD)),(pt1[0]-40,pt1[1]-20))







def calcDistRect(x4, y4, x3, y3, x2, y2, x, y):

    dist4 = round(dist.euclidean((x4, y4), (x3, y3)))
    dist3 = round(dist.euclidean((x3, y3), (x2, y2)))
    dist2 = round(dist.euclidean((x2, y2), (x, y)))
    dist1 = round(dist.euclidean((x, y), (x4, y4)))
    
    #print(dist1)
    cv2.putText(frame, str(dist1), (round(0.5 * x + 0.5 * x4), round(0.5 * y + 0.5 * y4)) , font, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, str(dist2), (round(0.5 * x2 + 0.5 * x), round(0.5 * y2 + 0.5 * y)) , font, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, str(dist4), (round(0.5 * x4 + 0.5 * x3), round(0.5 * y4 + 0.5 * y3)) , font, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, str(dist3), (round(0.5 * x3 + 0.5 * x2), round(0.5 * y3 + 0.5 * y2)) , font, 0.5, (0, 0, 0), 1)


    pt4 = x4, y4
    pt3 = x3, y3
    pt2 = x2, y2
    pt1 = x, y
    
    m2 = gradient(pt2,pt1)
    n2 = gradient(pt2,pt3)
    if m2 is not None and n2 is not None:
        angR2 = math.atan((n2-m2)/(1+(n2*m2)))
        angD2 = math.degrees(angR2)
        if math.isnan(angD2) is False:
            cv2.putText(frame, str(round(abs(angD2))), (pt2[0]-40,pt2[1]-20), font, 1, (0, 0, 0))        
            #print(round(abs(angD2)),(pt1[0]-40,pt1[1]-20))
    
    m3 = gradient(pt3,pt2)
    n3 = gradient(pt3,pt4)
    if m3 is not None and n3 is not None:
        angR3 = math.atan((n3-m3)/(1+(n3*m3)))
        angD3 = math.degrees(angR3)
        if math.isnan(angD3) is False:
            cv2.putText(frame, str(round(abs(angD3))), (pt3[0]-40,pt3[1]-20), font, 1, (0, 0, 0))        
            #print(round(abs(angD3)),(pt1[0]-40,pt1[1]-20))
    
    m4 = gradient(pt4,pt3)
    n4 = gradient(pt4,pt1)
    if m4 is not None and n4 is not None:
        angR4 = math.atan((n4-m4)/(1+(n4*m4)))
        angD4 = math.degrees(angR4)
        if math.isnan(angD4) is False:
            cv2.putText(frame, str(round(abs(angD4))), (pt4[0]-40,pt4[1]-20), font, 1, (0, 0, 0))        
            #print(round(abs(angD6)),(pt1[0]-40,pt1[1]-20))
    
    m = gradient(pt1,pt4)
    n = gradient(pt1,pt2)
    if m is not None and n is not None:
        angR = math.atan((n-m)/(1+(n*m)))
        angD = math.degrees(angR)
        if math.isnan(angD) is False:
            cv2.putText(frame, str(round(abs(angD))), (pt1[0]-40,pt1[1]-20), font, 1, (0, 0, 0))                
            #print(round(abs(angD)),(pt1[0]-40,pt1[1]-20))






def calcDistHex(x6, y6, x5, y5, x4, y4, x3, y3, x2, y2, x, y):
    dist1 = (dist.euclidean((x6, y6), (x5, y5)))
    dist2 = (dist.euclidean((x5, y5), (x4, y4)))
    dist3 = (dist.euclidean((x4, y4), (x3, y3)))
    dist4 = (dist.euclidean((x3, y3), (x2, y2)))
    dist5 = (dist.euclidean((x2, y2), (x, y)))
    dist6 = (dist.euclidean((x, y), (x6, y6)))
    
    #print(dist1)
    cv2.putText(frame, str(dist1), (round(0.5 * x6 + 0.5 * x5), round(0.5 * y6 + 0.5 * y5)) , font, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, str(dist2), (round(0.5 * x5 + 0.5 * x4), round(0.5 * y5 + 0.5 * y4)) , font, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, str(dist3), (round(0.5 * x4 + 0.5 * x3), round(0.5 * y4 + 0.5 * y3)) , font, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, str(dist4), (round(0.5 * x3 + 0.5 * x2), round(0.5 * y3 + 0.5 * y2)) , font, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, str(dist5), (round(0.5 * x2 + 0.5 * x), round(0.5 * y2 + 0.5 * y)) , font, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, str(dist6), (round(0.5 * x + 0.5 * x6), round(0.5 * y + 0.5 * y6)) , font, 0.5, (0, 0, 0), 1)

    pt6 = x6, y6
    pt5 = x5, y5
    pt4 = x4, y4
    pt3 = x3, y3
    pt2 = x2, y2
    pt1 = x, y
    
    m2 = gradient(pt2,pt1)
    n2 = gradient(pt2,pt3)
    if m2 is not None and n2 is not None:
        angR2 = math.atan((n2-m2)/(1+(n2*m2)))
        angD2 = math.degrees(angR2)
        if math.isnan(angD2) is False:
            cv2.putText(frame, str(round(abs(angD2))), (pt2[0]-40,pt2[1]-20), font, 1, (0, 0, 0))        
            #print(round(abs(angD2)),(pt1[0]-40,pt1[1]-20))
    
    m3 = gradient(pt3,pt2)
    n3 = gradient(pt3,pt4)
    if m3 is not None and n3 is not None:
        angR3 = math.atan((n3-m3)/(1+(n3*m3)))
        angD3 = math.degrees(angR3)
        if math.isnan(angD3) is False:
            cv2.putText(frame, str(round(abs(angD3))), (pt3[0]-40,pt3[1]-20), font, 1, (0, 0, 0))        
            #print(round(abs(angD3)),(pt1[0]-40,pt1[1]-20))
    
    m4 = gradient(pt4,pt3)
    n4 = gradient(pt4,pt5)
    if m4 is not None and n4 is not None:
        angR4 = math.atan((n4-m4)/(1+(n4*m4)))
        angD4 = math.degrees(angR4)
        if math.isnan(angD4) is False:
            cv2.putText(frame, str(round(abs(angD4))), (pt4[0]-40,pt4[1]-20), font, 1, (0, 0, 0))        
            #print(round(abs(angD4)),(pt1[0]-40,pt1[1]-20))
    
    m5 = gradient(pt5,pt4)
    n5 = gradient(pt5,pt6)
    if m5 is not None and n5 is not None:
        angR5 = math.atan((n5-m5)/(1+(n5*m5)))
        angD5 = math.degrees(angR5)
        if math.isnan(angD5) is False:
            cv2.putText(frame, str(round(abs(angD5))), (pt5[0]-40,pt5[1]-20), font, 1, (0, 0, 0))                
            #print(round(abs(angD5)),(pt1[0]-40,pt1[1]-20))
    
    m6 = gradient(pt6,pt5)
    n6 = gradient(pt6,pt1)
    if m6 is not None and n6 is not None:
        angR6 = math.atan((n6-m6)/(1+(n6*m6)))
        angD6 = math.degrees(angR6)
        if math.isnan(angD6) is False:
            cv2.putText(frame, str(round(abs(angD6))), (pt6[0]-40,pt6[1]-20), font, 1, (0, 0, 0))        
            #print(round(abs(angD6)),(pt1[0]-40,pt1[1]-20))
    
    m = gradient(pt1,pt6)
    n = gradient(pt1,pt2)
    if m is not None and n is not None:
        angR = math.atan((n-m)/(1+(n*m)))
        angD = math.degrees(angR)
        if math.isnan(angD) is False:
            cv2.putText(frame, str(round(abs(angD))), (pt1[0]-40,pt1[1]-20), font, 1, (0, 0, 0))                
            #print(round(abs(angD)),(pt1[0]-40,pt1[1]-20))
    if cv2.waitKey(1) == ord('h'):
            timestamp = int(time.time() * 10000)
            with open('dataset.csv', 'a', newline='') as dataset_file:
                dataset = csv.DictWriter(
                    dataset_file,
                    ["timestamp", "shape", "Side1", "Side2", "Side3", "Side4", "Side5", "Side6", "Perimeter", "Angle1", "Angle2", "Angle3", "Angle4", "Angle5", "Angle6", "AngleSum", "Error"]
                )
                dataset.writeheader()
                dataset.writerow({
                    "timestamp": timestamp,
                    "shape": "Hexagon",
                    "Side1": dist1,
                    "Side2": dist2,
                    "Side3": dist3,
                    "Side4": dist4,
                    "Side5": dist5,
                    "Side6": dist6,
                    "Perimeter": (dist1 + dist2 + dist3 + dist4 + dist5 + dist6),
                    "Angle1": angD,
                    "Angle2": angD2,
                    "Angle3": angD3,
                    "Angle4": angD4,
                    "Angle5": angD5,
                    "Angle6": angD6,
                    "AngleSum": (angD + angD2 + angD3 + angD4 + angD5 + angD6),
                    "Error": "To Do"

                })
                
    return dist1, dist2, dist3, dist4, dist5, dist6, angD, angD2, angD3, angD4, angD5, angD6;


cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)



font = cv2.FONT_HERSHEY_SIMPLEX
rectange = 0
triangle = 0
hexagon = 0

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 199, 5)
    #_, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)


        # Opencv 4.x.x
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for cnt in contours:
      
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if area > 20000:
            cv2.drawContours(frame, [approx], 0, (100, 0, 0), 2)

            if len(approx) == 6:
                #print(area)
                cv2.putText(frame, "Hexagon", (x, (y - 15)), font, 1, (0, 0, 0), 2)
            elif len(approx) == 4:
                cv2.putText(frame, "Quadrilateral", (x, (y - 15)), font, 1, (0, 0, 0), 2)
            elif len(approx) == 3:
                cv2.putText(frame, "Triangle", (x, (y - 15)), font, 1, (0, 0, 0), 2)
          
            elif 12 < len(approx) < 50:
                cv2.putText(frame, "Circle", (x, y), font, 1, (0, 0, 0), 2)

            n = approx.ravel()
            i = 0
            xn = 0
            yn = 0

            if len(approx) == 4:
                for j in n:
                    if(i % 2 == 0):
                        x4 = n[i - 6]
                        y4 = n[i - 5]

                        x3 = n[i - 4]
                        y3 = n[i - 3]
                        
                        x2 = n[i - 2]
                        y2 = n[i - 1]
                        
                        x = n[i]
                        y = n[i + 1]
                        
                        #print(x, y, x2, y2, x3, y3, x4, y4)
                        string = str(x) + " " + str(y)
                        cv2.circle(frame, (x, y), 2, (0,0,100), 2)                            
                        cv2.putText(frame, string, (x, y), font, 0.5, (138, 138, 54), 2)
                        calcDistRect(x4, y4, x3, y3, x2, y2, x, y)
                        
                                    # text on remaining co-ordinates.
            if len(approx) == 3:
                for j in n:
                    if(i % 2 == 0):

                        x3 = n[i - 4]
                        y3 = n[i - 3]
                        
                        x2 = n[i - 2]
                        y2 = n[i - 1]
                        
                        x = n[i]
                        y = n[i + 1]
                        
                        #print(x, y, x2, y2, x3, y3, x4, y4)
                        string = str(x) + " " + str(y)
                        cv2.circle(frame, (x, y), 2, (0,0,100), 2)                            
                        cv2.putText(frame, string, (x, y), font, 0.5, (138, 138, 54), 2)
                        calcDistTri(x3, y3, x2, y2, x, y)
                                    # text on remaining co-ordinates.

            if len(approx) == 6:
                for j in n:
                    if(i % 2 == 0):
                        x6 = n[i - 10]
                        y6 = n[i - 9]
                        
                        x5 = n[i - 8]
                        y5 = n[i - 7]

                        x4 = n[i - 6]
                        y4 = n[i - 5]

                        x3 = n[i - 4]
                        y3 = n[i - 3]
                        
                        x2 = n[i - 2]
                        y2 = n[i - 1]
                        
                        x = n[i]
                        y = n[i + 1]
                        
                        #print(x, y, x2, y2, x3, y3, x4, y4)
                        string = str(x) + " " + str(y)
                        cv2.circle(frame, (x, y), 2, (0,0,100), 2)                            
                        cv2.putText(frame, string, (x, y), font, 0.5, (138, 138, 54), 2)

                        calcDistHex(x6, y6, x5, y5, x4, y4, x3, y3, x2, y2, x, y)
                                
                                    # text on remaining co-ordinates.

                i = i + 1
    result.write(frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", threshold)

    key = cv2.waitKey(1)
                             
    if key == ord('q'):
        break
cap.release()
result.release()
cv2.destroyAllWindows()
