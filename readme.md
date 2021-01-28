# Kaggle: Digit Recognizer

<img style="margin-left:0" src="https://i.dlpng.com/static/png/1280814-best-25-number-writing-practice-ideas-on-pinterest-writing-writing-numbers-png-1280_752_preview.png" width="600px">

This is a solution for Kaggle Digit Recognizer competition. 
It's based on legendary MNIST dataset which contains 70,000 handwritten digits in 28x28 format.

The notebook includes 3 solutions:
- SVM on PCA (0.97610)
- MLP (0.97614)
- Ensamble of 10x CNN with soft voting prediction (0.99657, top 6%)

Competition Link: https://www.kaggle.com/c/digit-recognizer/overview

## Installation

Install project dependencies with Poetry:

```bash
poetry install
```

Install dataset:

```bash
cd data
kaggle competitions download -c digit-recognizer
```

Project requires Python 3.8.6 as it uses TensorFlow2.

## T-SNE Visualization

<img src="https://raw.githubusercontent.com/roma-glushko/kaggle-digit-recognizer/master/docs/tsne.png" width="600px" />
