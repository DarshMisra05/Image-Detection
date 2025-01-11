import cv2
import numpy as np
import csv
from datetime import datetime as dt
import os

# open CSV file. "if" statement checks to see if it needs to add data titles or not
csv_file = "blob_data.csv"
if not os.path.isfile(csv_file):
    with open(csv_file, mode = 'w', newline= '') as file:
        writer = csv.writer(file)
        writer.writerow(['Date and Time', 'x', 'y', 'diameter'])

# Preprocessing of the image
img = cv2.imread("prettyBuoy4.jpg")
img = cv2.blur(img, (10, 10))
img = cv2.resize(img, None, fx = 0.5, fy=0.5, interpolation= cv2.INTER_LINEAR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# These are the bounds for the colour we're looking at in HSV format
lower_greenA = np.array([0,50,50])
upper_greenA = np.array([60,255,255])

lower_greenB = np.array([170,50,50])
upper_greenB = np.array([180,255,255])

# Because red in HSV is 170 - 180 and 0 - 60 degrees, we need to `
maskA = cv2.inRange(hsv, lower_greenA, upper_greenA)
maskB = cv2.inRange(hsv, lower_greenB, upper_greenB)
mask = cv2.bitwise_or(maskA, maskB)

# perform bitwise and on the pre-processed image arrays using the mask
res = cv2.bitwise_and(img, img, mask=mask)

# Show the mask and the or'd image
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.imshow("Masked part of original image", res)
cv2.waitKey(0)

# Use edge detection to make a nice harsh contrast for the blob detector
res = cv2.Canny(mask, 50, 100, True)
cv2.imshow("Edges found", res)
cv2.waitKey(0)

# Set up of SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.7
params.filterByConvexity = False
params.filterByInertia = False
#params.minInertia = 0.1

# Detect blobs.
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(res)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Save keypoints information in CSV file
for keyPoint in keypoints:
    x = keyPoint.pt[0]
    y = keyPoint.pt[1]
    s = keyPoint.size
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Update the format to include both date and time
        writer.writerow([dt.now().strftime("%Y-%m-%d %H:%M:%S"), x, y, s])

print("Data appended successfully!")

# save blob data as CSV
csv_file = "pricing_data.csv"

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)