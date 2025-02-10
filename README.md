# Sign Language Detection using Deep Learning

## Overview
This project implements a deep learning model for recognizing Indian Sign Language (ISL) hand gestures. It uses convolutional neural networks (CNNs) trained on a dataset of sign images to classify different signs accurately. The model is built using TensorFlow and Keras and supports real-time sign prediction from images.

## Features
- Loads and preprocesses sign language images.
- Implements data augmentation for better generalization.
- Uses a CNN model with batch normalization, dropout, and pooling layers for improved accuracy.
- Trains the model using categorical cross-entropy loss and stochastic gradient descent (SGD) optimizer.
- Provides a function to predict sign language from new images.
- Visualizes model performance with accuracy/loss plots and a confusion matrix.
- Saves the trained model for later use.

## Dataset
The dataset used in this project consists of images from the **Indian Sign Language (ISL)** dataset available on Kaggle. The dataset is structured as follows:

```plaintext
/kaggle/input/indian-sign-language-isl/Indian/
    ├── A/
    ├── B/
    ├── C/
    ├── ...
    ├── Z/
```

Each subdirectory contains images representing a specific sign.

## Dependencies
Ensure you have the following libraries installed before running the code:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn pillow scikit-learn kaggle
```

## Model Architecture
The CNN model consists of:
- **3 Convolutional layers** with ReLU activation and batch normalization.
- **Max pooling layers** to downsample feature maps.
- **Dropout layers** to prevent overfitting.
- **Fully connected (Dense) layers** for classification.
- **Softmax output layer** to predict sign categories.

## Training
To train the model, run the script in a Kaggle or Colab environment. The model is trained with:
- **25 epochs**
- **Batch size of 32**
- **Learning rate reduction on validation accuracy plateau**

## Evaluation
After training, the model is evaluated using:
- **Accuracy and loss curves**
- **Confusion matrix for classification performance**
- **Validation accuracy score**

## Prediction
To predict a sign from an image:

```python
from PIL import Image
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load model
model = load_model("indianSignLanguage.h5")

# Function to predict sign
def predict_sign(image_path):
    img = Image.open(image_path).convert('RGB').resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]

# Example usage
image_path = "path/to/image.jpg"
predicted_sign = predict_sign(image_path)
print(f"The predicted sign is: {predicted_sign}")
```

## Model Saving & Loading
The trained model is saved as **indianSignLanguage.h5** for future inference:

```python
model.save("indianSignLanguage.h5")
```

To load the model for predictions:

```python
model = load_model("indianSignLanguage.h5")
```

## Results & Performance
- The model achieves high accuracy in recognizing ISL hand signs.
- Proper data augmentation helps in reducing overfitting.
- Using **SGD optimizer** with **learning rate reduction** improves convergence.

## Future Improvements
- **Expand dataset** to include more variations in lighting and backgrounds.
- **Implement real-time video-based recognition** using OpenCV.
- **Optimize the model** for deployment on mobile and edge devices.

