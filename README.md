# Neural Scene Representation
Train a GQN neural network and compare its representational structure to that of scene regions from human fMRI data.

## Generative Query Networks
GQN's are networks that extract invarient representations of scenes from images at different viewpoints. They essentially do this by making predictions about what a scene would look like from a novel viewpoint given the encoded representation of that scene. To solve this task, the encoded representation must contain information about the shape of space, the textures, the objects and their locations, etc.

Further information about the method can be found here https://science.sciencemag.org/content/360/6394/1204.full?ijkey=kGcNflzOLiIKQ&keytype=ref&siteid=sci. 

Model and training code come from https://github.com/wohlert/generative-query-network-pytorch.
