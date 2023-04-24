import cv2
import pandas as pd
import os
from a_GT_visualization import loadGroundTruth, createBoundingBox, createIdentityNumber
from b_myDetectionAlgorithm import loadImgDatasetPaths, removeBG, createMyTruth

def showBothWorking(all_images_path, groundTruthDf, myTruthDf, timeOfFrames=25, video_name='AlgorithmComparisson.mp4'):

    # Read the first image to get its dimensions
    img = cv2.imread(all_images_path[0])
    height, width, channels = img.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, 24.0, (width, height), isColor=True)

    # Loop through the images and display them one after another like a video
    for idx, img_path in enumerate(all_images_path):

        img = cv2.imread(img_path)

        # For ground truth
        df_GT = groundTruthDf
        # select all the boxes that shall appear in frame number idx+1
        boxes_GT = df_GT[df_GT["Frame number"] == idx+1]

        # For my algorithm
        df_myAnalysis = myTruthDf
        # select all the boxes that shall appear in frame number idx+1
        boxes_MT = df_myAnalysis[df_myAnalysis["Frame number"] == idx+1]
        
        # Two loops to create the bounding boxes
        for _, row_gt in boxes_GT.iterrows():
            xTopLeft_gt = row_gt["Bounding box left"]
            yTopLeft_gt = row_gt["Bounding box top"]
            width_gt = row_gt["Bounding box width"]
            height_gt = row_gt["Bounding box height"]
            id_gt = row_gt["Identity number"]
            img = createBoundingBox(img, xTopLeft_gt, yTopLeft_gt, width_gt, height_gt, (0, 255, 0))
            img = createIdentityNumber(img, xTopLeft_gt, yTopLeft_gt, width_gt, id_gt, (0, 255, 0))

        for _, row_mt in boxes_MT.iterrows():
            xTopLeft_mt = row_mt["Bounding box left"]
            yTopLeft_mt = row_mt["Bounding box top"]
            width_mt = row_mt["Bounding box width"]
            height_mt = row_mt["Bounding box height"]
            id_mt = row_mt["Identity number"]
            img = createBoundingBox(img, xTopLeft_mt, yTopLeft_mt, width_mt, height_mt, (0, 255, 255))
            img = createIdentityNumber(img, xTopLeft_mt, yTopLeft_mt, width_mt, id_mt, (0, 255, 255))

        cv2.imshow("Detection: Green (GT) and Yellow (my algorithm)", img)
        if cv2.waitKey(timeOfFrames) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        # Write the current frame to the video file
        out.write(img)

    cv2.destroyAllWindows()

    # Release the video writer
    out.release()
    print("Generated video: " + video_name)


if __name__ == "__main__":

    # Loading data form GT
    groundTruthDf = loadGroundTruth()

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
    showBothWorking(all_images_path, groundTruthDf, myTruthDf)