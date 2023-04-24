import cv2
import pandas as pd
import os
import re
from a_GT_visualization import loadGroundTruth, createBoundingBox, createIdentityNumber
from b_myDetectionAlgorithm import loadImgDatasetPaths, removeBG, createMyTruth
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn import metrics


def calculate_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0] + bb1[2], bb2[0] + bb2[2])
    y_bottom = min(bb1[1] + bb1[3], bb2[1] + bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

def calculate_success_plot(groundTruthDf, myTruthDf, thresholds=np.arange(0, 1.05, 0.05)):
    """
    Calculate the success plot using the Intersection over Union (IoU) measure.
    """
    print('\n\t -Calculating the IoU for the succes plot')

    num_frames = max(groundTruthDf['Frame number'].max(), myTruthDf['Frame number'].max())
    num_thresholds = len(thresholds)
    num_gt = len(groundTruthDf)
    num_pred = len(myTruthDf)


    # Calculate IoU for each detection and each ground truth
    iou_matrix = np.zeros((num_gt, num_pred))

    for i, row_gt in groundTruthDf.iterrows():
        gt_box = row_gt[['Bounding box left', 'Bounding box top', 'Bounding box width', 'Bounding box height']].values
        # Iterate on all the rows of myTruthDf that have the same frame number as the row in groundTruthDf
        for j, row_pred in myTruthDf[myTruthDf['Frame number'] == row_gt['Frame number']].iterrows():
            pred_box = row_pred[['Bounding box left', 'Bounding box top', 'Bounding box width', 'Bounding box height']].values
            test = calculate_iou(gt_box, pred_box)

            i_idx = groundTruthDf.index.get_loc(i)
            j_idx = myTruthDf.index.get_loc(j)
            iou_matrix[i_idx, j_idx] = calculate_iou(gt_box, pred_box)

    # Calculate success rates for each threshold
    success_rates = np.zeros((num_thresholds,))

    for i in range(num_thresholds):
        threshold = thresholds[i]

        num_successes = 0
        for j in range(num_frames):

            gt_indices = np.where(groundTruthDf['Frame number']==j+1)[0]
            if len(gt_indices) == 0:
                continue
            # 1D array with each iou associated with each ground truth box
            max_iou_for_gt = iou_matrix[gt_indices,:].max(axis=1)
            # Number of yellow boxes that have IoU greater than threshold
            num_gt_successes = (max_iou_for_gt >= threshold).sum()
            num_successes += num_gt_successes

        success_rates[i] = float(num_successes) / float(num_gt)


    # Compute AUC
    auc_score = auc(thresholds, success_rates)

    # Plot success plot
    plt.plot(thresholds, success_rates, label='AUC = {:.3f}'.format(auc_score))
    plt.title('Success plot')
    plt.xlabel('Overlap threshold (IoU)')
    plt.ylabel('Success rate')
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    plt.legend()
    plt.savefig('success_plot.png')
    plt.show()



def calculate_detection_results(groundTruthDf, myTruthDf, thresholds=np.arange(0, 1.05, 0.05)):
    """
    Calculate the detection results using the Intersection over Union (IoU) measure.
    """
    num_frames = max(groundTruthDf['Frame number'].max(), myTruthDf['Frame number'].max())
    num_thresholds = len(thresholds)
    num_gt = len(groundTruthDf)
    num_pred = len(myTruthDf)

    # Calculate IoU for each detection and each ground truth
    iou_matrix = np.zeros((num_gt, num_pred))

    for i, row_gt in groundTruthDf.iterrows():
        gt_box = row_gt[['Bounding box left', 'Bounding box top', 'Bounding box width', 'Bounding box height']].values
        # Iterate on all the rows of myTruthDf that have the same frame number as the row in groundTruthDf
        for j, row_pred in myTruthDf[myTruthDf['Frame number'] == row_gt['Frame number']].iterrows():
            pred_box = row_pred[['Bounding box left', 'Bounding box top', 'Bounding box width', 'Bounding box height']].values
            i_idx = groundTruthDf.index.get_loc(i)
            j_idx = myTruthDf.index.get_loc(j)
            iou_matrix[i_idx, j_idx] = calculate_iou(gt_box, pred_box)

    # Calculate detection results for each threshold
    detection_results = np.zeros((num_thresholds, num_frames, 3))

    for i in range(num_thresholds):
        threshold = thresholds[i]

        for j in range(num_frames):

            gt_indices = np.where(groundTruthDf['Frame number']==j+1)[0]
            if len(gt_indices) == 0:
                continue
            # 1D array (collumn) with any ground truth box associated with each predicted box
            # if one element is 0 it means that there are no predicted boxes for that particular ground truth box.
            max_iou_for_gt = iou_matrix[gt_indices,:].max(axis=1)
            # Number of ground truth boxes that have IoU greater than threshold (True positives)
            # If it was detected twice no problem: it enters as detected
            num_gt_successes = (max_iou_for_gt >= threshold).sum()

            pred_indices = np.where(myTruthDf['Frame number']==j+1)[0]
            if len(pred_indices) == 0:
                num_fp = 0
            else:
                # 1D array (collumn) with any ground truth box associated with each predicted box
                max_iou_for_pred = iou_matrix[:,pred_indices].max(axis=0)
                # Number of yellow boxes (collumns) that have IoU less than threshold
                num_fp = (max_iou_for_pred < threshold).sum()

            # i is the threshold and j the frame. The np.array inside has the FP, FN and TP
            detection_results[i, j] = np.array([num_fp, len(gt_indices)-num_gt_successes, num_gt_successes])

    # Initialize arrays to store totals
    false_positives_totals = np.zeros(num_thresholds)
    false_negatives_totals = np.zeros(num_thresholds)
    true_positives_totals = np.zeros(num_thresholds)

    for i in range(num_thresholds):

        # Calculate totals for each type of detection for this threshold
        false_positives_totals[i] = detection_results[i,:,0].sum()
        false_negatives_totals[i] = detection_results[i,:,1].sum()
        true_positives_totals[i] = detection_results[i,:,2].sum()


    # Divide the thresholds into two halves
    mid_threshold = num_thresholds // 2

    fig, axs = plt.subplots(nrows=mid_threshold, ncols=3, figsize=(11, 6))

    for i in range(mid_threshold):
        threshold = thresholds[i]

        # Plot bar graph
        fp = detection_results[i,:,0]
        fn = detection_results[i,:,1]
        tp = detection_results[i,:,2]
        axs[i,0].bar(np.arange(num_frames), fp, color='red', label='False positives')
        axs[i,0].bar(np.arange(num_frames), fn, bottom=fp, color='yellow', label='False negatives')
        axs[i,0].bar(np.arange(num_frames), tp, bottom=fp+fn, color='green', label='True positives')
        axs[i,0].set_title(f'Detection results for IoU threshold = {threshold:.2f}', fontsize=9)
        axs[i,0].set_xlabel('Frame number', fontsize=7)
        axs[i,0].set_ylabel('Number of detections', fontsize=7)
        axs[i,0].legend(loc='upper right', fontsize=5)

        # Set font size for tick labels
        axs[i,0].tick_params(axis='both', which='major', labelsize=6)


        # Plot pie chart
        labels = ['True positives', 'False positives', 'False negatives']
        sizes = true_positives_totals[i], false_negatives_totals[i], false_positives_totals[i]
        colors = ['green', 'yellow', 'red']
        textprops = {'fontsize': 7}
        wedges, _, autotexts = axs[i,1].pie(sizes, colors=colors, labels=labels, textprops=textprops, radius=1.3, autopct='%1.1f%%')
        axs[i,1].set_title(f'Detection results for IoU threshold = {threshold:.2f}', fontsize=9)

    
        # Set font size for percentage values in pie chart
        for autotext in autotexts:
            autotext.set_fontsize(6)

        # Plot horizontal bar graph
        axs[i,2].barh(['False positives', 'False negatives', 'True positives'], [false_positives_totals[i], false_negatives_totals[i], true_positives_totals[i]], color=['red', 'yellow', 'green'])
        axs[i,2].set_title(f'Total detection results for IoU threshold = {threshold:.2f}', fontsize=9)
        axs[i,2].set_xlabel('Number of detections', fontsize=7)
        axs[i,2].set_ylabel('Type of detection', fontsize=7)

        # Set font size for tick labels
        axs[i,2].tick_params(axis='both', which='major', labelsize=6)

    plt.tight_layout()

    fig2, axs2 = plt.subplots(nrows=num_thresholds-mid_threshold, ncols=3, figsize=(11, 6))
    for i in range(num_thresholds-mid_threshold):
        j = i + mid_threshold
        threshold = thresholds[j]

        # Plot bar graph
        fp = detection_results[j,:,0]
        fn = detection_results[j,:,1]
        tp = detection_results[j,:,2]
        axs2[i,0].bar(np.arange(num_frames), fp, color='red', label='False positives')
        axs2[i,0].bar(np.arange(num_frames), fn, bottom=fp, color='yellow', label='False negatives')
        axs2[i,0].bar(np.arange(num_frames), tp, bottom=fp+fn, color='green', label='True positives')
        axs2[i,0].set_title(f'Detection results for IoU threshold = {threshold:.2f}', fontsize=9)
        axs2[i,0].set_xlabel('Frame number', fontsize=7)
        axs2[i,0].set_ylabel('Number of detections', fontsize=7)
        axs2[i,0].legend(loc='upper right', fontsize=5)

        # Set font size for tick labels
        axs2[i,0].tick_params(axis='both', which='major', labelsize=6)

        # Plot pie chart
        labels = ['True positives', 'False positives', 'False negatives']
        sizes = true_positives_totals[j], false_negatives_totals[j], false_positives_totals[j]
        colors = ['green', 'yellow', 'red']
        textprops = {'fontsize': 6}
        wedges, _, autotexts = axs2[i,1].pie(sizes, colors=colors, labels=labels, textprops=textprops, radius=1.3, autopct='%1.1f%%')
        axs2[i,1].set_title(f'Detection results for IoU threshold = {threshold:.2f}', fontsize=9)

        # Set font size for percentage values in pie chart
        for autotext in autotexts:
            autotext.set_fontsize(5)

        # Plot horizontal bar graph
        axs2[i,2].barh(['False positives', 'False negatives', 'True positives'], [false_positives_totals[j], false_negatives_totals[j], true_positives_totals[j]], color=['red', 'yellow', 'green'])
        axs2[i,2].set_title(f'Total detection results for IoU threshold = {threshold:.2f}', fontsize=9)
        axs2[i,2].set_xlabel('Number of detections', fontsize=7)
        axs2[i,2].set_ylabel('Type of detection', fontsize=7)

        # Set font size for tick labels
        axs2[i,2].tick_params(axis='both', which='major', labelsize=6)

    plt.tight_layout()

    fig.savefig('EvaluationPerformance - low thresholds.png', dpi=300)
    fig2.savefig('EvaluationPerformance - high thresholds.png', dpi=300)

    plt.show()




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

    # Check if the file exists
    if os.path.isfile(filename):
        # If the file exists, read it in
        print("This analysis was already done")
        myTruthDf = pd.read_csv(filename)
    else:
        # If the file does not exist, run removeBG and createMyTruth and save the output to a file
        imagesWithoutBG = removeBG(all_images_path, minThreshold=minThreshold, openingK=openingK, dilationK=dilationK, closingK=closingK, smallClosingK=smallClosingK, minArea=minArea)
        myTruthDf = createMyTruth(imagesWithoutBG)
        myTruthDf.to_csv(filename, index=False)

    # Plot the Sucess Plot
    calculate_success_plot(groundTruthDf, myTruthDf, thresholds=np.arange(0, 1.01, 0.1))
    
    # FP, FN and TP
    calculate_detection_results(groundTruthDf, myTruthDf, thresholds=np.arange(0, 1.01, 0.2))

    



'''
    # Code to test in just some images, it may only work if it frames_to_select starts from 1 ascending to the final value (range(1, final_frame)) 
    #frames_to_select = [10, 100, 500]
    #frames_to_select = range(1,100)
    # select the first 10 frames from groundTruthDf
    #groundTruthDf_selected = groundTruthDf.loc[groundTruthDf['Frame number'].isin(frames_to_select)]

    # select the first 10 frames from myTruthDf
    #myTruthDf_selected = myTruthDf.loc[myTruthDf['Frame number'].isin(frames_to_select)]

    # path to onlt selected images
    #selected_image_paths = [all_images_path[frame_number-1] for frame_number in frames_to_select]



    #calculate_success_plot(groundTruthDf, myTruthDf, thresholds=np.arange(0, 1, 0.1))
    #print(groundTruthDf_selected)
    #print(myTruthDf_selected)
    
    # Sucess plot
    #plot_success(groundTruthDf_selected, myTruthDf_selected)
'''
