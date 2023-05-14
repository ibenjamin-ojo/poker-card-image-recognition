# Poker Card Image Recognition

This is a machine learning project that uses convolutional neural networks (CNN) to classify poker cards based on their type. The goal of the project is to develop a model that can accurately identify a given card image as one of 53 different categories of cards.

## Dataset

The dataset used for this project is the [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification), which contains over 8000 images of poker cards in JPG format. The dataset is divided into three sets: training (7624 images), validation (265 images), and testing (265 images). The images are of size 224 x 224 pixels with three color channels.

## Methodology

The project follows a standard machine learning workflow, including the following steps:

1. **Data Collection:** The dataset was downloaded from Kaggle and examined to gain insights into the data.

2. **Data Preprocessing:** The images were preprocessed by resizing them to a standard size and converting them into an array format suitable for feeding into our CNN model.

3. **Data Augmentation:** Data augmentation techniques were used to increase the size of the dataset and improve the robustness of the model. Techniques used include image rotation, flipping, and zooming.

4. **Model Building:** A CNN model was built using Keras with TensorFlow backend, and trained using the preprocessed dataset. Different architectures, hyperparameters, and optimization algorithms were experimented with to achieve optimal performance.

5. **Model Evaluation:** The performance of the model was evaluated using various metrics such as accuracy, precision, recall, and F1 score. The results were visualized using a confusion matrix and classification report.

6. **Model Deployment:** The trained model was deployed to make predictions on new, unseen poker card images. The predictions were visualized. 

## Getting Started

To run this project, you will need to have the following tools and libraries installed:

- Python 3.8+
- Keras
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn

You can run the project on your local machine or in a Jupyter notebook environment such as Google Colab.

1. Clone this repository to your local machine:

```
git clone https://github.com/yourusername/poker-card-image-recognition.git
```

2. Download the dataset from Kaggle and extract it to the `data` folder in the project directory.

3. Open the Jupyter notebook `poker-card-image-recognition.ipynb` and run the cells in order.

4. Once the model is trained, you can use it to make predictions on new poker card images.

## Credits

- The dataset used in this project is the [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification) from Kaggle, created by Greg Piosenka.
