# README

In this project we made an attempt to auto-generate the caption of a given image.

We built a pipeline consisting of an encoder-decoder architecture. The encoder portion constitutes of a pre-trained Resnet CNN
to extract the features from the images and an embedding layer to get a convenient and concise feature-vector.

The feature vector obtained as output from the encoder part was thrown as an input into the decoder part which constitues of 
a Long Short Term Memory cell (a kind of RNN) and an embedding layer which outputs a vector of a certain length which will be
compared by the pre-written caption labels of the training images.

*Please refer to the folder project2 2

