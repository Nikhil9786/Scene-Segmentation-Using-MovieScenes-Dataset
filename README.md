# Scene-Segmentation-Using-MovieScenes-Dataset

The main aim of this challenge was to predict the probabilities of shot boundaries, in other words also scene boundary.

## Data
I was provided with the MovieScenes [dataset](https://drive.google.com/file/d/1oZSOkd4lFmbY205VKQ9aPv1Hz3T_-N6e/view).This dataset contains 64 .pkl files.
The data contains:
1. Movie level: Contains IMDB ID
2. Shot-level: Contains four features; 'place','cast','action', and 'audio'. These features are preprocessed and encoded as two-dimensional tensors, with the first dimension indicating the number of shots within a movie, and the second dimension specifying feature vectors, in this case, 2048, 512, 512, and 512, respectively.
3. Scene-level:
    * Ground Truth(‘scene_transition_boundary_ground_truth’) - a boolean vector labeling scene transition boundaries.
    * Preliminary scene transition prediction (‘scene_transition_boundary_prediction’) - sample outputs

## Algorithm
Function make_predictions reads and concat all the pkl files into one. Later on convert it into dataframe which is split into features and predictors.
for better training and testing, the dataset is split into two with 80% being used for training and 20% used for testing purpose. For better prections, different regression models were compared. I used regression model instead of any other say classification model because for prediction model when we need a number as an output(in our case predicted probabilities), it is a good idea to use regression model. Of all the models I compared, Logistic Regression model gave the best predicted probabilities and alsp mAP and Miou.

To calculate mAP and Miou, I took reference from the this Github [Repository](https://github.com/eluv-io/elv-ml-challenge)

## Code
**Requirements**

Run ```pip install -r requirements.txt```

1. Make a folder to store result pkl files, in my case Results folder.
2. Run on command prompt ```python Scene-Segmentation.py data_dir Results``` to traing the model and get predicted probabilties in pkl files.
3. To test the model using [evaluate_sceneseg.py](https://github.com/Nikhil9786/Scene-Segmentation-Using-MovieScenes-Dataset/blob/main/evaluate_sceneseg.py) run ```python evaluate_sceneseg.py Results``` on command prompt.

## Result Metrics
Following pic snippet shows all the metrics I got from training and testing this model using Logistic Regression.

<p align="center">
  <img width="460" height="300" src="https://github.com/Nikhil9786/Scene-Segmentation-Using-MovieScenes-Dataset/blob/main/Result.JPG">
</p>

## Refrences
1. Baraldi, Lorenzo, et al. “A Deep Siamese Network for Scene Detection in Broadcast Videos.” [arXiv.org, 2015]( https://arxiv.org/abs/1510.08893)
2. Rao, Anyi, et al. “A Local-to-Global Approach to Multi-modal Movie Scene Segmentation.” [arXiv.org, 2020]( https://arxiv.org/abs/2004.02678)
