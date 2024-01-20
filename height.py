import cv2
import numpy as np

img = cv2.imread('data/Layer/201211301200_rx3_60min87bin.png')

h,w = img.shape[0],img.shape[1]
print(h,w)
arr = [0]*h
for i in range(h):
    sum = np.array([0,0,0])
    for j in range(w):
        sum+=np.array(img[i][j])
    arr[i] = sum/w
    
# newimage = img
# for i in range(h):
#     for j in range(w):
#         newimage[i][j]=arr[i]
target_colors = [(26, 0, 0),(80,18,0),(100,96,1),(71,150,0),(1,135,15),(0,48,120),(0,24,180),(1,108,223),(0,166,237),(249,226,0)]
val = [1,2,3,4,5,6,7,8,9,10]
color_threshold = 100
matching_positions = []
newarr = [-1]*h
for k in range(len(target_colors)):
    for i in range(h):
        distance = np.linalg.norm(np.array(target_colors[k]) - np.array(arr[i]))
        if distance<=color_threshold:
            newarr[i]=val[k]
p= newarr.index(max(newarr))
newarr.reverse()
q = newarr.index(max(newarr))
newarr.reverse()
p = p + (h - q)
p=int(p/2)
range_of_height = 50
h_of_layer = 75 + ((50/h)*p)
print(h_of_layer)
y = p  # Replace with the desired height
color = (0, 255, 0)  # Green line
cv2.line(img, (0, y), (img.shape[1], y), color, thickness=2)
cv2.imshow('Image with Line', img)
cv2.waitKey(0)  # Wait for a key press to close the window
#   for y, row in enumerate(img):
#       for x, pixel in enumerate(row):
#           distance = np.linalg.norm(np.array(target_color) - np.array(pixel))
#           if distance <= color_threshold:
#               matching_positions.append((ind,x, y))
#   # data.append(matching_positions)
#   counts.append(len(matching_positions))
#   ind+=1
#   print(matching_positions)
# cv2.imshow("newimage",newimage)
# cv2.waitKey(0)