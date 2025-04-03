import cv2
import os 

mainFolder = 'images'
myFolders = os.listdir(mainFolder)

for folder in myFolders :
    path = mainFolder + '/' + folder
    images = []
    myList = os.listdir(path)
    
    for imgN in myList :
        curImg = cv2.imread(path + '/' + imgN)
        curImg = cv2.resize(curImg, (0, 0), None, 0.2, 0.2)
        images.append(curImg)

    stitcher = cv2.Stitcher_create()
    (status, result) = stitcher.stitch(images)
    
    if (status == cv2.Stitcher_OK):
        print("Panorama created successfully.")
        cv2.imshow(folder, result)
        cv2.waitKey(1)
    else:
        print("Panorama creation failed.")
        
cv2.waitKey(0)