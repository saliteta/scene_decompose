# Ground Truth, Metrics, and General Viewer

## Ground Truth
To generate ground truth, we just need to have their camera pose in COLMAP format, and then we generate a 2D ones map, and we use splat feature solver to lift the attention map to Gaussian, and the default value (Outside image) would be -1.

## Dinstribution Alignment Score
Two torch tensor input, shape is N (Gaussian Number), output is one scaler:
Input: 
1. Scale Three torch tensors to -1 to 1. One is the predicted value, and one is the groud truth value
2. (Dot product of these two, and divide by N) +1  /2, range from -1 to 1


## General Viewer
- Support the Ground Truth Visualization if Enabled
- Should Support Many Different Feature Query Method
- Should Support Current Hierachical Viewer

