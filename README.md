# trajectory-analysis-CV-IST

## Goal and data used

The goal of this work is to develop an **algorithm capable detecting the location of pedestrians** as a way to obtain the performed trajectories and other important data to help analyse the **situation** and the **effectiveness** of our algorithm. All this with focus on using **handcrafted features**.

For this, it is used the Dataset S2: People Tracking, available on this website https://cs.binghamton.edu/~mrldata/pets2009
Specifically, it is used the dataset S2.L1 (View001). This means that isolated pedestrians are considered with little interactions between them.

## Getting Started

To run this project, you'll need to install the python packages listed in the file *requirements.txt* . Once you have those set up, you can start running the python files to do the tracking of pedestrians.

Also, you can use this repository to analyse other datasets, you just need to put the images inside the folder *PutImageDatasetHere*, the ground truth in the folder *PutGroundTruthHere* and possibly change the following functions:
- ``loadGroundTruth()``: if your ground truth does not have the same columns as the one in this rep.
- ``removeBG()``: change manually its parameters until the morphological operations are working the way you want it.

## Files Included

- ``a_GT_visualization.py``: visualizes the bounding boxes for the ground truth of the scene.
- ``b_myDetectionAlgorithm.csv``: visualizes the bounding boxes for this algorithm, the trajectories of pedestrians a dinamic heatmap and a static heatmap.
- ``c_detection_comparison.py``: shows both bounding boxes so you can compare your algorithm.
- ``d_EvaluationPerformance.py``: shows a sucess plot and also PN, FN and TP for different thresholds values for the IoU of this algorithm and the ground truth.
- ``e_savingHeatmapVideos.py``: code for saving the videos of the heatmaps.
- ``myTruthDf-15-4-1-6-3-200.csv``: contains the whole analysis of this algorithm (each time you change the parameters of the morphological operations and run the code a new file like this will be created)

## How to use

Each python file is responsible for solving a certain challenge, you just need to run the ``.py`` file you want and it will show you plots, videos and save them accordingly.

## Expected results

After you run all the python files, you will see **videos that show the tracking** working:

![Algorithm Comparisson - Example](My%20analysis%20results/AlgorithmComparissonExample.png)

And also data of about the **quality** of the performance.
This is a success plot that shows how well fitted this algorithms bounding boxes were, compared to the ground truth:

![Sucess plot](My%20analysis%20results/success_plot.png)

And these graphs show the numbers of True Positives, False Positives and False Negatives according to the threshold (a 0.4 threshold means that only bounding boxes that had an IoU of 40% or more were considered correct):
![Algorithm Performance (low IoU)](My%20analysis%20results/EvaluationPerformance%20-%20low%20thresholds.png)
![Algorithm Performance (high IoU)](My%20analysis%20results/EvaluationPerformance%20-%20high%20thresholds.png)
