## Transformer based Model with EfficientNet as Backbone

This model is created to play around with Transformers. The Model is trained on a Kaggle Dataset: https://www.kaggle.com/c/plant-pathology-2021-fgvc8
The problem is to detect if a leaf has one (or multiple) sickness signs, or if it is healthy.

### Deep Learning Model

The Model has a Backbone Model, which takes an image as input and extracts the data. This Model is a pre trained (on ImageNet) EfficientNet model.
The output of this model is further reshaped and by adding positional encoding passed to a Visual Transformer based model.

The model is trained with Adam with weight decay optimizer and uses binary crossentropy as well as tripplet loss to classifly the objects in the image.