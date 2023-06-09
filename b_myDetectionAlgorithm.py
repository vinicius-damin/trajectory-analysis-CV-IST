import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from a_GT_visualization import createBoundingBox, createIdentityNumber


# Loads all the names of the frames from the dataset
def loadImgDatasetPaths():

    print('\n\t -Selecting images')

    image_folder = "PutImageDatasetHere/View_001"
    imagesPaths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
    imagesPaths.sort()

    return imagesPaths



# Receives the path to all images and returns the images without the background by using a median filter
def removeBG(pathToImages, minThreshold=15, openingK=4, dilationK=1, closingK=6, smallClosingK=3, minArea=200):

    print('\n\t -Applying morphological transformations images')

    print('Loading images...')
    # Load all images
    images = [cv2.imread(imgPath) for imgPath in pathToImages]

    print('Computing median to remove background...')
    # Compute median image (background) by subsampling all the images and return a grayscale image
    median = cv2.cvtColor(np.median(images[::10], axis=0).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Initialize list to store binary images without background
    imagesWithoutBG = []

    print('Applying morphological transformation on all images...')
    # Loop through all images
    for img in images:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Subtract the median from the grayscale image to obtain the foreground
        foreground = cv2.absdiff(gray, median)
   
        # Threshold the foreground to obtain a binary image
        thresholded = cv2.threshold(foreground, minThreshold, 255, cv2.THRESH_BINARY)[1]

        # Apply opening to remove small objects or noise from binary image
        kernel = np.ones((openingK, openingK),np.uint8)
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

        # Dilation to fill gap between head and body
        kernel = np.ones((dilationK,dilationK),np.uint8)
        dilated = cv2.dilate(opened, kernel)
        dilated = cv2.dilate(dilated, kernel)

        # Perform a closing operation to fill in big gaps in the objects
        kernel = np.ones((closingK,closingK),np.uint8)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        # Perform many closing operations to fill in small gaps in the objects
        kernel = np.ones((smallClosingK,smallClosingK),np.uint8)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)     

        imagesWithoutBG.append(closed)
    print('Morphological transformations applied.')
    return imagesWithoutBG



# Creates a dataframe that has all the data to create the bounding boxes on all images
def createMyTruth(imagesWithoutBG):

    print('\n\t -Creating dataframe with the data from the computer vision analysis')

    # Define the column names for the dataframe
    columns = ["Frame number", "Identity number", "Bounding box left", "Bounding box top",
               "Bounding box width", "Bounding box height", "Area", "Centroid"]

    # Initialize empty list for empty dataframe for tracking data
    dfMyTruth = pd.DataFrame(columns=columns)

    # Initialize dictionary to store centroids for each identity number
    centroids_dict = {}

    print('Iterating through every binary image...')
    # Loop through each binary image in the list
    for frame_num, binary_img in enumerate(imagesWithoutBG):

        # Perform connected component analysis on the binary image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

        # Loop through each connected component in the binary image
        for i in range(1, num_labels):

            # Extract stats for this component
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Remove noise by avaliating only big regions
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


                # Add tracking data to the my truth dataframe
                new_line = pd.DataFrame({
                        "Frame number": frame_num+1, #as the txt, it starts at 1
                        "Identity number": identity_num,
                        "Bounding box left": left,
                        "Bounding box top": top,
                        "Bounding box width": width,
                        "Bounding box height": height,
                        "Area": area,
                        "xCentroid": cx,
                        "yCentroid": cy
                }, index=[0])
                dfMyTruth = pd.concat([dfMyTruth, new_line], ignore_index=True)

    print('Data collected.')

    return dfMyTruth



def showMTworking(imagesPath, myAnalysisDf):

    print('\n\t -Starting the presentation of the results')
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

            img = createBoundingBox(img, xTopLeft, yTopLeft, width, height, (0, 255, 255))
            img = createIdentityNumber(img, xTopLeft, yTopLeft, width, id, (0, 255, 255))

        cv2.imshow("My detection algorithm", img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    print('Finished.')
    cv2.destroyAllWindows()
    return



def showMTTrajectories(all_images_path, myTruthDf):
    # Read the first image to get its dimensions
    img = cv2.imread(all_images_path[0])
    height, width, channels = img.shape

    # Initialize the canvas with transparent background
    canvas = np.zeros((height, width, 4), dtype=np.uint8)

    # Set the alpha channel of the canvas to 50 (semi-transparent)
    canvas[:, :, 3] = 50

    # Define a dictionary to map identity numbers to colors
    id_color_dict = {}

    # Loop over all the images
    for i, img_path in enumerate(all_images_path):
        # Read the image
        img = cv2.imread(img_path)

        # Get the rows for the current frame
        df = myTruthDf[myTruthDf['Frame number'] == i]

        # Loop over the rows
        for _, row in df.iterrows():
            # Get the centroid coordinates
            x = int(row['xCentroid'])
            y = int(row['yCentroid'])

            # Get the identity number
            identity_number = row['Identity number']

            # Check if the identity number is already in the dictionary
            if identity_number in id_color_dict:
                # If it is, use the existing color
                color = id_color_dict[identity_number]
            else:
                # If not, generate a random color and add it to the dictionary
                color = np.random.randint(0, 255, size=3)
                color = tuple(map(int, color))
                id_color_dict[identity_number] = color

            # Draw the circle on the canvas with the corresponding color
            cv2.circle(canvas, (x, y), 5, color + (50,), -1)

        # Show the canvas overlaid on the image
        cv2.imshow('Trajectories', cv2.addWeighted(img, 0.7, canvas[:, :, :3], 0.3, 0))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()



def generateDinamicHeatmap(all_images_path, myTruthDf, sigma=20):
    # Loop over all the images
    for i, img_path in enumerate(all_images_path):
        # Read the image
        img = cv2.imread(img_path)

        # Get the rows for the current frame
        df = myTruthDf[myTruthDf['Frame number'] == i]

        # Read the first image to get its dimensions
        height, width, channels = img.shape

        # Initialize the heatmap with zeros
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Loop over the rows
        for _, row in df.iterrows():
            # Get the centroid coordinates
            x = int(row['xCentroid'])
            y = int(row['yCentroid'])

            # Set the standard deviation of the Gaussian kernel
            sigma = sigma

            # Calculate the distance from the current point to all other points
            xv, yv = np.meshgrid(np.arange(width), np.arange(height))
            distance = np.sqrt((xv - x)**2 + (yv - y)**2)

            # Calculate the weights for each pixel based on the distance
            weights = np.exp(-distance**2 / (2 * sigma**2))

            # Update the heatmap with the weighted values
            heatmap += weights

        # Normalize the heatmap to a range of [0, 1]

        if np.max(heatmap) == 0:
            heatmap_norm = heatmap
        else:
            heatmap_norm = heatmap / np.max(heatmap)

        # Convert the heatmap to a 3-channel image
        heatmap = cv2.cvtColor(np.uint8(heatmap_norm * 255), cv2.COLOR_GRAY2BGR)

        # Resize the heatmap to match the size of the original image
        heatmap = cv2.resize(heatmap, (width, height))

        # Add the heatmap on top of the original image
        heatmap_img = cv2.addWeighted(img, 0.2, heatmap, 0.8, 0)

        # Show the heatmap image
        cv2.imshow('Dinamic heatmap', heatmap_img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()



def generateSimpleHeatmap(all_images_path, myTruthDf, size=10):

    print("\n\t -Calculating simple heatmap")

    # Read the first image to get its dimensions
    img = cv2.imread(all_images_path[0])
    height, width, channels = img.shape

    # Initialize the heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Loop over all the images
    for i, img_path in enumerate(all_images_path):
        # Read the image
        img = cv2.imread(img_path)

        # Get the rows for the current frame
        df = myTruthDf[myTruthDf['Frame number'] == i]

        # Loop over the rows
        for _, row in df.iterrows():
            # Get the centroid coordinates
            x = int(row['xCentroid'])
            y = int(row['yCentroid'])

            # Add to the heatmap around the given coordinates
            heatmap[max(0, y-size):min(height, y+size), max(0, x-size):min(width, x+size)] += 1

    # Calculate the percentage of time spent in each area
    total_time = len(all_images_path)
    heatmap_percentage = heatmap / total_time * 100

    # Plot the heatmap
    plt.imshow(heatmap_percentage, cmap='inferno')

    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('% of time spent in area')

    # Title
    plt.title("Occupancy heatmap (whole scene)")

    # Show the plot
    plt.show()
    print("Plotted")
    


def generateGaussianHeatmap(all_images_path, myTruthDf, sigma=20):

    print("\n\t -Calculating Gaussian heatmap (whole scene)")

    # Read the first image to get its dimensions
    img = cv2.imread(all_images_path[0])
    height, width, channels = img.shape

    # Initialize the heatmap with zeros
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Loop over all the images
    for i, img_path in enumerate(all_images_path):
        # Read the image
        img = cv2.imread(img_path)

        # Get the rows for the current frame
        df = myTruthDf[myTruthDf['Frame number'] == i]

        # Loop over the rows
        for _, row in df.iterrows():
            # Get the centroid coordinates
            x = int(row['xCentroid'])
            y = int(row['yCentroid'])

            # Set the standard deviation of the Gaussian kernel
            sigma = sigma

            # Calculate the distance from the current point to all other points
            xv, yv = np.meshgrid(np.arange(width), np.arange(height))
            distance = np.sqrt((xv - x)**2 + (yv - y)**2)

            # Calculate the weights for each pixel based on the distance
            weights = np.exp(-distance**2 / (2 * sigma**2))

            # Update the heatmap with the weighted values
            heatmap += weights

    # Calculate the percentage of time spent in each area
    total_time = len(all_images_path)
    heatmap_percentage = heatmap / total_time * 100

    # Plot the heatmap
    plt.imshow(heatmap_percentage, cmap='inferno')

    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('% of time spent in area')

    # Title
    plt.title("Occupancy heatmap (whole scene)")

    # Show the plot
    plt.show()
    print("Plotted")



if __name__ == "__main__":

    # Getting images
    all_images_path = loadImgDatasetPaths()

    # Define the parameters for removeBG function
    minThreshold = 15
    openingK = 4
    dilationK = 1
    closingK = 6
    smallClosingK = 3
    minArea = 200

    # Define the filename
    filename = f"myTruthDf-{minThreshold}-{openingK}-{dilationK}-{closingK}-{smallClosingK}-{minArea}.csv"

    # Check if the simulation was already done, if so, load the data
    if os.path.isfile(filename):
        # If the file exists, read it in
        print("This analysis was already done")
        myTruthDf = pd.read_csv(filename)

    # Otherwise, load run the algorithm, create the data and save it to a .csv
    else:
        # If the file does not exist, run removeBG and createMyTruth and save the output to a file
        imagesWithoutBG = removeBG(all_images_path, minThreshold=minThreshold, openingK=openingK, dilationK=dilationK, closingK=closingK, smallClosingK=smallClosingK, minArea=minArea)
        myTruthDf = createMyTruth(imagesWithoutBG)
        myTruthDf.to_csv(filename, index=False)

    # Presenting results
    showMTworking(all_images_path, myTruthDf)

    # Shows the trajectories dinamically
    showMTTrajectories(all_images_path, myTruthDf)

    # Show dinamic heatmap of each frame
    generateDinamicHeatmap(all_images_path, myTruthDf, sigma=50)

    # Plot the a simple static heatmap (10*10 calculations per person, n per frame)
    generateSimpleHeatmap(all_images_path, myTruthDf, size=10)

    # Plot the static heatmap using gaussian metric (w*h calculations per person, n*w*h per frame)
    generateGaussianHeatmap(all_images_path, myTruthDf, sigma=20)





'''
Code if I want to see a specific result. Just need to change imagesWithoutBG.append(closed) if needed.

        for img in imagesWithoutBG:
            cv2.imshow("Image without background", img)
            if cv2.waitKey(25) and 0xFF == ord('q'):
                cv2.destroyAllWindows()
               break

       cv2.destroyAllWindows()
'''