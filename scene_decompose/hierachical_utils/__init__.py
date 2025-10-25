"""
We first constructing tree abstraction for the hierarchical primitive.
Our hierarchical primitive should be compatible with current query system.

Hierachical primitive are essentially several layers of Nodes.
One node is a set of torch tensors. (e.g. mean, quat, scale, opacity, color, feature)
While cross layer connection is following a binary tree structure.

For example: 
Layer 0: only one node, that means we have one set of torch tensors
Layer 1: two nodes, that means we have two sets of torch tensors
Layer 2: four nodes, that means we have four sets of torch tensors
...
Layer N: 2^N nodes, that means we have 2^N sets of torch tensors

For layer 0, node 0, it can be represented as a None
For layer 1, node 0, it can be represented as a 0, node 1 can be represented as a 1

Layer 0: None
Layer 1: 0, 1
Layer 2: 00, 01, 10, 11
Layer 3: 000, 001, 010, 011, 100, 101, 110, 111

If I want to know who are the decendent of node 00, I can simply follow the binary tree structure.
In layer 3, node 00 has the following decendents: 000, 001
In layer 4, node 00 has the following decendents: 0000, 0001, 0010, 0011
In layer N, node I in layer L has the following decendents: 
I<<(N-L) to (I+1)<<(N-L) - 1


Notice that we are creating a seperate query system in the 2DX3D repo, and we want to 
pass that query system class as a parameter to the HierachicalViewer class.

The query system do the following work:

1. Given a query text, it will return the index of the primitive to query # Implemented in Splat Feature Solver
2. Given a list of index, it will return the primitive to query # Will be implemented soon
3. Given torch tensor and range, return which one is the most similar to the query # Will be implemented soon
"""