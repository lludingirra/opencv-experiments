import cv2  # Import the OpenCV library for image processing.
import pickle  # Import the pickle library for serializing and de-serializing Python objects.


width, height = 107, 48  # Define the width and height of each parking spot rectangle.

try:
    with open('CarParkPos', 'rb') as f:  # Open the 'CarParkPos' file in read binary mode ('rb'). This file is assumed to contain parking spot positions.
        posList = pickle.load(f)  # Load the parking spot positions from the file and assign them to the posList variable.
except FileNotFoundError:  # If the 'CarParkPos' file is not found, this block will be executed.
    posList = []  # Initialize posList as an empty list, meaning no parking spots have been selected yet.
    print("CarParkPos file not found. Creating a new one.")  # Print a message to the user.
except Exception as e:  # If any other error occurs during file reading, this block will handle it.
    print(f"An error occurred: {e}")  # Print the error message.
    posList = []  # Initialize posList as an empty list.

# posList = []  # This line was moved inside the try block. Now posList is always defined.


def mouseClick(events, x, y, flags, params):
    """
    Function to handle mouse click events.

    Args:
        events: The type of mouse event (e.g., left-click, right-click).
        x: The x-coordinate of the mouse click.
        y: The y-coordinate of the mouse click.
        flags: Any flags passed by OpenCV.
        params: Any extra parameters passed by OpenCV.
    """

    global posList  # Declare posList as a global variable. This is necessary to modify it within the function.
    if events == cv2.EVENT_LBUTTONDOWN:  # If the left mouse button is clicked:
        posList.append((x, y))  # Append the clicked point's coordinates (x, y) to the posList. This adds a new parking spot.
        try:
            with open('CarParkPos', 'wb') as f:  # Open the 'CarParkPos' file in write binary mode ('wb').
                pickle.dump(posList, f)  # Serialize and save the updated posList to the file. This saves the selected parking spot positions.
        except Exception as e:  # If an error occurs during file writing, this block will handle it.
            print(f"An error occurred while saving: {e}")  # Print the error message.

    if events == cv2.EVENT_RBUTTONDOWN:  # If the right mouse button is clicked:
        for i, pos in enumerate(posList):  # Iterate through the posList to find the clicked parking spot. i is the index, and pos is the coordinate.
            x1, y1 = pos  # Get the coordinates of the current parking spot.

            if x1 < x < x1 + width and y1 < y < y1 + height:  # If the clicked point is within the rectangle of the current parking spot:
                posList.pop(i)  # Remove the parking spot from the list using its index.
                try:
                    with open('CarParkPos', 'wb') as f:  # Open the 'CarParkPos' file in write binary mode.
                        pickle.dump(posList, f)  # Serialize and save the updated posList to the file.
                except Exception as e:  # If an error occurs during file writing, this block will handle it.
                    print(f"Error saving after deletion: {e}")  # Print the error message.
                break  # Exit the loop.  Important:  This prevents deleting multiple spots if they overlap.

while True:  # Enter an infinite loop to continuously process and display the image.

    img = cv2.imread("carParkImg.png")  # Read the image 'carParkImg.png' using OpenCV and store it in the img variable.
    if img is None:  # If the image loading fails (e.g., file not found):
        print("carParkImg.png could not be loaded! Please make sure the file exists.")  # Print an error message.
        break  # Exit the loop and the program will terminate.

    for pos in posList:  # Iterate through the posList, which contains the coordinates of selected parking spots.
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (0, 0, 255),
                      2)  # Draw a rectangle on the image for each parking spot.
        # img: The image to draw on.
        # pos: The starting point of the rectangle.
        # (pos[0] + width, pos[1] + height): The ending point of the rectangle.
        # (0, 0, 255): The color of the rectangle (red in BGR format).
        # 2: The thickness of the rectangle's border.

    cv2.imshow("Image", img)  # Display the image with the drawn rectangles in a window named "Image".
    cv2.setMouseCallback("Image", mouseClick)  # Set the mouse click callback function for the "Image" window to the mouseClick function.
    if cv2.waitKey(1) & 0xFF == ord(
            'q'):  # Wait for a key event for 1 millisecond.  If the 'q' key is pressed, exit the loop.
        break  # Exit the loop if the 'q' key is pressed.

cv2.destroyAllWindows()  # Close all OpenCV windows.
