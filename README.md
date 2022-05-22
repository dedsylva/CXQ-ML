# CXQ-ML

-------
A comparison of Classical and Quantum Machine Learning Approach for Neural Networks in <b>binary image classification</b>.

## Running
```bash
# classical networks
set MODEL=IMAGENET && python classical/scratch/classical.py
set MODEL=MNIST && python classical/scratch/classical.py

# quantum networks (prerocess=1 passes through first quantum layer)
set MODEL=IMAGENET && set PREPROCESS=1 && python quantum/scratch/qml.py
```

## Objective

-------
How performant is one over the other? We will try to answer this by designing two types of neural networks for each approach (Classical and Quantum). 
 - From scratch
      - training neural network with no prior knowledge
 - Using Transfer Learning
      - using knowledge with networks trained in Imagenet classification

### Datasets

-------
The datasets that will be used are:
 - MNIST (60k Images)
    - http://yann.lecun.com/exdb/mnist/
 - Bees and Ants taken from Imagenet (240 Images)
    - https://download.pytorch.org/tutorial/hymenoptera_data.zip
 - COVID-19 X-Ray (317 Images)
    - https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset
 - Blood Cells for Malaria prediction (27.5k Images) 
    - https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria


### Tools

-------
- Classical ML
 - Tensorflow

- QML
 - Pennylane


#### TODO
- MNIST Dataset
   - Classical [X]
   - Quantum [X]
- Bees and Ants Dataset
   - Classical [X]
   - Quantum []
- COVID-19 X-Rays Dataset
   - Classical []
   - Quantum []
- Malaria Blood Cells Dataset
   - Classical []
   - Quantum []
