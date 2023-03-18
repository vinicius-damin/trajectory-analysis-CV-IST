import cv2
import os
import pandas as pd

# Loads all the names of the frames from the dataset
def loadImgDatasetPaths():

    image_folder = "PutImageDatasetHere/View_001"
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
    images.sort()

    return images

# Loads the Ground Truth (.txt file)
# Loads the Ground Truth (.txt file)
def loadGroundTruth():

    dir_path = "PutTheGroundTruthHere"

    # Define the column names
    columns = ["Frame number", "Identity number", "Bounding box left", "Bounding box top",
            "Bounding box width", "Bounding box height", "Confidence score", "x", "y", "z"]

    # Loop over all the files in the directory
    for filename in os.listdir(dir_path):
        # Set the path of the file
        file_path = os.path.join(dir_path, filename)
        # Read the file into a pandas DataFrame
        df = pd.read_csv(file_path, header=None, names=columns)
        # Add the DataFrame to the list

    return df

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

def showGTworking(all_images_path, groundTruthDf):
    # Loop through the images and display them one after another like a video
    for idx, img_path in enumerate(all_images_path):

        img = cv2.imread(img_path)
        df = groundTruthDf
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
        

if __name__ == "__main__":
    all_images_path = loadImgDatasetPaths()
    groundTruthDf = loadGroundTruth()
    showGTworking(all_images_path, groundTruthDf)


'''
    # Create a black background image
    #img = cv2.imread(r"PutImageDatasetHere\View_001\frame_0001.jpg")
    print(all_images_path[0])
    img = cv2.imread(all_images_path[0])


    # Draw a yellow bounding box on the image
    imgWithBox = createBoundingBox(img, 499,158,31.03,75.17)
    imgWithBox = createBoundingBox(imgWithBox, 258,219,32.913,88.702)
    imgWithBox = createBoundingBox(imgWithBox, 633,242,42.338,81.074)

    # Show the original image and the image with the bounding box
    cv2.imshow("Original Image", img)
    cv2.imshow("Image with Bounding Box", imgWithBox)
    cv2.waitKey(0)


#TAVA FUNCIONANADO
'''
