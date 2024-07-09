import cv2
import numpy as np
import pickle

# Dimensions of each parking space rectangle
rectW, rectH = 107, 48

# Open the video file
cap = cv2.VideoCapture('carPark.mp4')


# Load parking positions from a pickle file
with open('CarParkPos.unknown', 'rb') as f:
    posList = pickle.load(f)

# Initialize the frame counter
frame_counter = 0

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

def check(imgPro):

    # Check parking spaces and draw rectangles to indicate free or occupied spaces.
    spaceCount = 0
    for pos in posList:
        x, y = pos

        # Crop the processed image to the area of the parking space
        crop = imgPro[y:y+rectH, x:x+rectW]

        # Count the number of non-zero (white) pixels in the cropped image
        count = cv2.countNonZero(crop)
        if count < 900:
            # If less than 900 white pixels, the space is considered free
            spaceCount += 1
            color = (0, 255, 0) # Green color for free space
            thick = 5
        else:
            # Otherwise, the space is considered occupied
            color = (0, 0, 255) # Red rectangle for occupied space
            thick = 2
        # Draw the rectangle on the original image
        cv2.rectangle(img, pos, (x + rectW, y + rectH), color, thick)

    # Draw a filled rectangle to display the count of free spaces
    cv2.rectangle(img, (45, 30), (250, 75), (180, 0, 180), -1)
    # Display the count of free spaces on the image
    cv2.putText(img, f'Free: {spaceCount}/{len(posList)}', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

while True:
    # Read a frame from the video
    ret, img = cap.read()
    if not ret:
        break

    # Reset the frame counter if the end of the video is reached
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the grayscale image
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    # Apply adaptive thresholding to the blurred image
    Thre = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    # Apply median blur to the thresholded image
    blur = cv2.medianBlur(Thre, 5)
    # Create a kernel for dilation
    kernel = np.ones((3, 3), np.uint8)
    # Apply dilation to the blurred image
    dilate = cv2.dilate(blur, kernel, iterations=1)
    # Check the parking spaces in the processed image
    check(dilate)

    out.write(img)  # Write the processed frame to the output video file
    cv2.waitKey(10) # Introduce a small delay to allow the image to be displayed

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
