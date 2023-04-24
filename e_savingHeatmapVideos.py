import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.cm as cm
from a_GT_visualization import loadGroundTruth, createBoundingBox, createIdentityNumber
from b_myDetectionAlgorithm import loadImgDatasetPaths, removeBG, createMyTruth

def generateGaussianHeatmapVideo(all_images_path, myTruthDf, sigma=20, video_name='StaticHeatmap.mp4'):
    print("\n\t -Calculating Gaussian heatmap (whole scene)")

    # Read the first image to get its dimensions
    img = cv2.imread(all_images_path[0])
    height, width, channels = img.shape

    # Initialize the heatmap with zeros
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, 24.0, (width, height), isColor=True)

    # Loop over all the images
    for i, img_path in enumerate(all_images_path):
        print(f'Analysing frame number: {i+1}')
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


        # Normalize the heatmap to a range of [0, 1]
        if np.max(heatmap) == 0:
            heatmap_norm = heatmap
        else:
            heatmap_norm = heatmap / np.max(heatmap)


        # Apply the inferno colormap
        heatmap_norm = (heatmap_norm * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_INFERNO)

        # Add the heatmap on top of the original image
        heatmap_img = cv2.addWeighted(img, 0.1, heatmap_color, 0.9, 0)

        # Write the current frame to the video file
        out.write(heatmap_img)

    # Release the video writer
    out.release()
    print("Generated video: " + video_name)



def generateDinamicHeatmap(all_images_path, myTruthDf, sigma=20, video_name='dinamicHeatmap.mp4'):
    
    # Read the image
    img = cv2.imread(all_images_path[0])

    # Read the first image to get its dimensions
    height, width, channels = img.shape
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, 24.0, (width, height), isColor=True)

    # Loop over all the images
    for i, img_path in enumerate(all_images_path):
        print(f'Analysing frame number: {i+1}')

        # Read the image
        img = cv2.imread(img_path)

        # Get the rows for the current frame
        df = myTruthDf[myTruthDf['Frame number'] == i]

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

        # Apply the inferno colormap
        heatmap_norm = (heatmap_norm * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_INFERNO)

        # Resize the heatmap to match the size of the original image
        heatmap_color = cv2.resize(heatmap_color, (width, height))

        # Add the heatmap on top of the original image
        heatmap_img = cv2.addWeighted(img, 0.2, heatmap_color, 0.8, 0)

        # Write the current frame to the video file
        out.write(heatmap_img)

    # Release the video writer and close all windows
    out.release()
    cv2.destroyAllWindows()


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


    # Save video of the static heatmap using gaussian metric (w*h calculations per person, n*w*h per frame)
    generateGaussianHeatmapVideo(all_images_path, myTruthDf, sigma=20)

    # Show dinamic heatmap of each frame
    #generateDinamicHeatmap(all_images_path, myTruthDf, sigma=50)



