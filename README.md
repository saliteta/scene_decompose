# Scene Decomposition

## Motivation and Problem Definition
- We want to have further understanding of current 3D representation
- Visual Place Recognition is a good start
- The definition would be the following: 
Input: 3D Representation and one or multiple 2D images
Output: Location in 3D representation, where those 2D images been taken


## Current Progress
- SAMOpenCLIP, mission impossible, takes over 12 hours for one scene pre-processing
- Splat Feature Lifting: Previous Job Splat Feature Solver
- Scene Decomposition through naive HDBSCAN (Failed)
- Scene Visualization: LOOKS OK
- Low Frequency Feature (CLIP Per Image + Image Retrival, working, not good)
- Direct Head Classification (Looks OK)


## TO DO
- For Lifted Gaussian, according to the PCA-TREE establish merge procedure and establish a hierachical feature
- Establish 3D Node Features and 2D Patchify Image Feature Cross Attention Mechanism
- Experiments and Explaination