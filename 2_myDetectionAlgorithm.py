import cv2
import os
import pandas as pd
import numpy as np

# Loads all the names of the frames from the dataset
def loadImgDatasetPaths():

    image_folder = "PutImageDatasetHere/View_001"
    imagesPaths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
    imagesPaths.sort()

    return imagesPaths


# Receives the path to all images and returns the images without the background by using a median filter
def removeBG(pathToImages, minThreshold=15, openingK=4, dilationK=1, closingK=6, smallClosingK=3, minArea=200):

    # Load all images
    images = [cv2.imread(imgPath) for imgPath in pathToImages]

    # Compute median image (background) and return a grayscale image
    median = cv2.cvtColor(np.median(images, axis=0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    cv2.imshow("median img", median)

    # Initialize list to store binary images without background
    imagesWithoutBG = []

    # Loop through all images
    for img in images:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray img", gray)
        # Subtract the median from the grayscale image to obtain the foreground
        foreground = cv2.absdiff(gray, median)
        #cv2.imshow("foreground img", foreground)
   
        # Threshold the foreground to obtain a binary image
        thresholded = cv2.threshold(foreground, minThreshold, 255, cv2.THRESH_BINARY)[1]
        #cv2.imshow("thresholded img 30", thresholded)   

        # Apply opening to remove small objects or noise from binary image
        kernel = np.ones((openingK, openingK),np.uint8)
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        #cv2.imshow("opened 4", opened)  

        # Dilation to fill gap between head and body
        kernel = np.ones((dilationK,dilationK),np.uint8)
        dilated = cv2.dilate(opened, kernel)
        dilated = cv2.dilate(dilated, kernel)

        # Perform a closing operation to fill in big gaps in the objects
        kernel = np.ones((closingK,closingK),np.uint8)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        # Perform closing operations to fill in small gaps in the objects
        kernel = np.ones((smallClosingK,smallClosingK),np.uint8)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
        
        #cv2.imshow("closed img 3", closed)

        imagesWithoutBG.append(closed)

    return imagesWithoutBG

# Creates a dataframe that has all the data to create the bounding boxes on all images
def createMyTruth(imagesWithoutBG):
    # Define the column names for the dataframe
    columns = ["Frame number", "Identity number", "Bounding box left", "Bounding box top",
               "Bounding box width", "Bounding box height", "Area"]

    # Initialize empty list for empty dataframe for tracking data
    dfMyTruth = pd.DataFrame(columns=columns)

    # Initialize dictionary to store centroids for each identity number
    centroids_dict = {}

    # Loop through each binary image in the list
    for frame_num, binary_img in enumerate(imagesWithoutBG):

        # Perform connected component analysis on the binary image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

        # Loop through each connected component
        for i in range(1, num_labels):

            # Extract stats for this component
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Check if the area is greater than or equal to 200
            if area >= 200:
                

                # Get centroid coordinates
                cx, cy = centroids[i]

                # Check if this is a new or existing identity (if is new pedestrian or an existing one)
                existing_ids = list(centroids_dict.keys())
                if len(existing_ids) == 0:
                    # If there are no existing pedestrians, assign identity number 1 to this component
                    identity_num = 1
                    centroids_dict[identity_num] = (cx, cy)
                else:
                    # If there are existing pedestrians, find the closest one and assign it to this component
                    min_dist = float('inf')
                    min_id = None
                    for identity_num, centroid in centroids_dict.items():
                        dist = np.sqrt((cx - centroid[0])**2 + (cy - centroid[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            min_id = identity_num
                    if min_dist > 50:
                        # If the closest identity is too far away, create a new one
                        max_id = max(existing_ids)
                        identity_num = max_id + 1
                        centroids_dict[identity_num] = (cx, cy)
                    else:
                        # Otherwise, assign the closest identity to this component
                        identity_num = min_id
                        centroids_dict[identity_num] = (cx, cy)

                # Add tracking data to the dataframe
                dfMyTruth = dfMyTruth.append(pd.DataFrame({
                    "Frame number": frame_num+1, #as the txt, it starts at 1
                    "Identity number": identity_num,
                    "Bounding box left": left,
                    "Bounding box top": top,
                    "Bounding box width": width,
                    "Bounding box height": height,
                    "Area": area
                }, index=[0]), ignore_index=True)

    return dfMyTruth

# Plots a bounding box on a certain frame
def createBoundingBox(originalImg, xTopLeft, yTopLeft, width, height):
    imgWithBox = originalImg.copy()  # make a copy of the original image
    # Pixel index must be int
    xTopLeft = int(xTopLeft)
    yTopLeft = int(yTopLeft)
    width = int(width)
    height = int(height)
    cv2.rectangle(imgWithBox, (xTopLeft, yTopLeft), (xTopLeft + width, yTopLeft + height), (0, 255, 255), 2)
    return imgWithBox

def createIdentityNumber(originalImg, xTopLeft, yTopLeft, width, id):
    # Pixel index must be int
    xTopLeft = int(xTopLeft)
    yTopLeft = int(yTopLeft)
    width = int(width)

    imgWithID = originalImg
    text = str(int(id))
    textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    textX = xTopLeft + (width - textSize[0]) // 2
    textY = yTopLeft - textSize[1] + 5
    cv2.putText(imgWithID, text, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return imgWithID

def showMTworking(imagesPath, myAnalysisDf):
    # Loop through the images and display them one after another like a video
    for idx, img_path in enumerate(imagesPath):

        img = cv2.imread(img_path)
        df = myAnalysisDf
        # select all the boxes that shall appear in frame number idx+1
        boxes = df[df["Frame number"] == idx+1]
        
        for _, row in boxes.iterrows():
            xTopLeft = row["Bounding box left"]
            yTopLeft = row["Bounding box top"]
            width = row["Bounding box width"]
            height = row["Bounding box height"]
            id = row["Identity number"]

            img = createBoundingBox(img, xTopLeft, yTopLeft, width, height)
            img = createIdentityNumber(img, xTopLeft, yTopLeft, width, id)

        cv2.imshow("image", img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    all_images_path = loadImgDatasetPaths()


    imagesWithoutBG = removeBG(all_images_path)

#    for img in imagesWithoutBG:
#        cv2.imshow("Image without background", img)
#        if cv2.waitKey(25) and 0xFF == ord('q'):
#            cv2.destroyAllWindows()
#           break
#
#   cv2.destroyAllWindows()

    myTruthDf = createMyTruth(imagesWithoutBG)

    #print(dfMyTruth)

    showMTworking(all_images_path, myTruthDf)