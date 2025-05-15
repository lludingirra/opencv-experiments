import cv2  # Import OpenCV for video and image processing
import pickle  # Import pickle for serializing and de-serializing Python object structures
import numpy as np  # Import numpy for numerical operations, especially for handling arrays
import cvzone  # Import the cvzone library, which provides computer vision helper functions (like drawing rectangles with text)

cap = cv2.VideoCapture('carPark.mp4')  # Initialize video capture from the 'carPark.mp4' video file.  This opens the video stream.

with open('CarParkPos', 'rb') as f:  # Open the file 'CarParkPos' in read binary mode ('rb') to load saved parking space positions.
    posList = pickle.load(f)  # Load the list of parking space positions (x, y coordinates) from the file.  posList will contain a list of tuples.

width, height = 107, 48  # Define the width and height of each parking space rectangle.  These dimensions are used to draw the rectangles.

def checkParkingSpcae(imgPro):
    """
    This function checks the availability of parking spaces in a processed image.

    Args:
        imgPro:  A processed grayscale image (e.g., thresholded, dilated) representing the parking area.
                 It should highlight occupied areas with high pixel values.
    """

    spaceCounter = 0  # Initialize a counter for the number of free parking spaces.

    for pos in posList:  # Iterate through each parking space position in the posList.
        x, y = pos  # Get the x and y coordinates of the top-left corner of the parking space.

        imgCrop = imgPro[y:y + height,
                  x:x + width]  # Crop the processed image to isolate the region corresponding to the current parking space.
        # cv2.imshow(str(x*y), imgCrop)  #  Display the cropped image (commented out, likely for debugging).  The window title is the product of x and y.
        count = cv2.countNonZero(
            imgCrop)  # Count the number of non-zero pixels in the cropped image.  This approximates the amount of "activity" in the space.
        #  A higher count means the space is more likely to be occupied.

        if count < 900:  # If the non-zero pixel count is below a threshold (900), the space is considered free.
            color = (0, 255, 0)  # Set the color to green (BGR) to indicate a free space.
            thickness = 5  # Set the thickness of the rectangle border.
            spaceCounter += 1  # Increment the free space counter.

        else:  # If the non-zero pixel count is above the threshold, the space is considered occupied.
            color = (0, 0, 255)  # Set the color to red (BGR) to indicate an occupied space.
            thickness = 2  # Set the thickness of the rectangle border.

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color,
                      thickness)  # Draw a rectangle on the original image to represent the parking space.
        #  img: The original image.
        #  pos: The top-left corner of the rectangle.
        #  (pos[0] + width, pos[1] + height): The bottom-right corner of the rectangle.
        #  color: The color of the rectangle (green or red).
        #  thickness: The thickness of the rectangle's border.
        cvzone.putTextRect(img, str(count), (x, y + height - 3),  # Put text on the image to display the non-zero pixel count.
                           scale=1, thickness=2,
                           offset=0, colorR=color)
        #  img:  The image to draw on
        #  str(count):  The text to display (the non-zero pixel count).
        #  (x, y + height -3):  The position of the text.
        #  scale, thickness, offset:  Text properties.
        #  colorR:  The color of the text (same as the rectangle).

    cvzone.putTextRect(img, f'Free : {spaceCounter}/{len(posList)}', (100, 50),  # Display the total number of free spaces and total spaces.
                       scale=3, thickness=5,
                       offset=20, colorR=(0, 255, 0))
    #  img: The image to draw on.
    #  f'Free : {spaceCounter}/{len(posList)}':  The text string.
    #  (100, 50):  The position of the text.
    #  scale, thickness, offset: Text properties.
    #  colorR:  The color of the text (green).


while True:  # Main loop to continuously process video frames.
    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(
            cv2.CAP_PROP_FRAME_COUNT):  # Check if the current frame position has reached the end of the video.
        cap.set(cv2.CAP_PROP_POS_FRAMES,
                0)  # If the end is reached, reset the video frame position to the beginning (loop the video).

    success, img = cap.read()  # Read the next frame from the video capture.
    #  success:  Boolean indicating if the frame was read successfully.
    #  img:  The captured frame as a NumPy array.

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale.  This simplifies image processing.
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3),
                               1)  # Apply Gaussian blur to the grayscale image.  This reduces noise and smooths the image.
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25,
                                         16)  # Apply adaptive thresholding to the blurred image.  This creates a binary image where pixels are either black or white.
    #   imgBlur:  The source image.
    #   255:  The maximum pixel value.
    #   cv2.ADAPTIVE_THRESH_GAUSSIAN_C:  Adaptive thresholding method that uses a Gaussian window.
    #   cv2.THRESH_BINARY_INV:  Thresholding type (pixels below the threshold are set to 255, others to 0).
    #   25:  Block size.
    #   16:  Constant subtracted from the mean.

    imgMedian = cv2.medianBlur(imgThreshold,
                               5)  # Apply median blur to the thresholded image.  This further reduces noise, especially "salt-and-pepper" noise.
    kernel = np.ones((3, 3),
                     np.uint8)  # Create a 3x3 kernel of ones.  This is used for dilation.
    imgDilate = cv2.dilate(imgMedian, kernel,
                           iterations=1)  # Dilate the median blurred image.  This expands the white regions and fills in small holes.

    checkParkingSpcae(
        imgDilate)  # Call the function to check parking space availability using the processed image.
    if not success:  # If reading a frame was unsuccessful (e.g., end of video stream).
        break  # Exit the loop.

    cv2.imshow("Image", img)  # Display the original image with the parking space markings in a window named "Image".

    if cv2.waitKey(10) & 0xFF == ord(
            'q'):  # Wait for a key event for 10 milliseconds.  If the 'q' key is pressed, exit the loop.
        break  # Exit the loop if the 'q' key is pressed.

cap.release()  # Release the video capture object to free resources.
cv2.destroyAllWindows()  # Close all OpenCV windows.
