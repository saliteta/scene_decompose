# Hierachical Gaussian

## Motivation and Problem Definition
- We want to have further understanding of current 3D representation
- Visual Place Recognition is a good start
- The definition would be the following: 
Input: 3D Representation and one or multiple 2D images
Output: Location in 3D representation, where those 2D images been taken


## Preliminaries
- One need to have per Gaussian Features
- One can follow our pipeline here or do one's own
- We recommend to follow: https://splat-distiller.pages.dev/ to add your Gaussian Features. But we essentially just need a N,C tensor, where N is the number of Gaussian, and C is the number of channels

### 1. Train Gaussian: 
- Follow gsplat, train one's own Gaussian 

### 2. Lifting (Currently only support JAFAR hight dimensional dinov3 model)
```
python distill_wrapper.py -data.dir ${ur data dir contain colmap info} --distill.ckpt {trained Gaussian CKPT} --model.model-path {high dimensional dinov3 model pth}
```

### 3. Constructing Hierachical GS
```
python construct_hierachical_gs -c {ckpt path} -f {feature location} -o {output path}
```

### 4. Visualization
```
python hierachical_viewer.py -s {the output GS}
```


## TO DO
- [x] For Lifted Gaussian, according to the PCA-TREE establish merge procedure and establish a hierachical feature
- [x] Establish Hierachical Gaussian Loader, and Gaussian Viewer using gsplat_ext
- [x] Establish the Gradio Visualization Tab for Image Features and blending with gsplat_ext
- [x] Establish the node traversing visualization
- [x] Establishing a universal Query System that can fit into viewer and can pipelining
- [x] Establish 3D Node Features and 2D Patchify Image Feature Cross Attention Mechanism
- [ ] Experiments and Explaination