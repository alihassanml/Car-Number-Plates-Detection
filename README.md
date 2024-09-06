# Car Number Plates Detection

This repository contains a deep learning project for detecting car number plates using object detection techniques and the InceptionResNetV2 model. The project is built on a dataset of car images with annotated bounding boxes around the number plates. The trained model is capable of detecting the location of number plates in images and extracting the region of interest.

## Dataset

The dataset used in this project is available on Kaggle:

[Car Number Plate Dataset](https://www.kaggle.com/alihassanml/car-number-plate)

To download the dataset, you can use the Kaggle API:

```bash
!kaggle datasets download -d alihassanml/car-number-plate
```

## Project Overview

The project involves the following steps:
1. **Data Loading and Preprocessing**:
    - The car images and corresponding XML files with bounding box annotations are loaded.
    - The bounding box coordinates (xmin, xmax, ymin, ymax) are extracted from the XML files, and the images are normalized and resized to 224x224 pixels.

2. **Model Architecture**:
    - The pre-trained InceptionResNetV2 model is used for feature extraction, with the top layers removed.
    - A custom head model is added with Dense layers to predict the bounding box coordinates for number plates in the images.
    - The model is trained using the Mean Squared Error (MSE) loss function to minimize the difference between predicted and actual bounding box coordinates.

3. **Training**:
    - The dataset is split into training and test sets using `train_test_split`.
    - The model is trained for 200 epochs using the Adam optimizer with a learning rate of 1e-4.
    - TensorBoard is used for monitoring the training process.

4. **Evaluation**:
    - The model’s training and validation loss are plotted after each training phase to analyze the model performance.

## Requirements

To run this project, you need to install the following libraries:

```bash
pip install pandas numpy matplotlib opencv-python tensorflow keras scikit-learn kaggle
```

Additionally, make sure to set up the Kaggle API to download the dataset.

## Model Summary

The model used in this project is based on the InceptionResNetV2 architecture, which is known for its high performance on image classification and detection tasks. The model's layers include:

- **InceptionResNetV2**: Pre-trained on ImageNet, used for feature extraction.
- **Custom Dense Layers**: Added to predict the bounding box coordinates for the detected car number plates.

```python
# Model Architecture Summary
model.summary()
```

## Training

The model was trained using the following configuration:

- **Optimizer**: Adam
- **Loss**: Mean Squared Error (MSE)
- **Batch Size**: 5
- **Epochs**: 200
- **Learning Rate**: 1e-4

The training process includes an initial training phase of 100 epochs, followed by an additional 100 epochs starting from epoch 101.

```python
# Model Training
history = model.fit(X_train, y_train, batch_size=5, epochs=200, validation_data=(X_test, y_test), callbacks=[TensorBoard('logs')])
```

## Results

The loss and validation loss during training are plotted to assess the performance of the model over time:

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()
```

## Model Saving

After training, the model is saved both locally and to Google Drive for future use:

```python
# Save the model locally and to Google Drive
model.save('model.h5')
keras.models.save_model(model, '/content/drive/MyDrive/Yoloy Model/my_model.keras')
```

## How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/alihassanml/Car-Number-Plates-Detection.git
    ```

2. Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset:
    ```bash
    !kaggle datasets download -d alihassanml/car-number-plate
    ```

4. Run the notebook in Google Colab or any Jupyter environment.

## Future Work

- **Live Detection**: Implement real-time car number plate detection using OpenCV and a webcam.
- **OCR Integration**: Use Tesseract or other Optical Character Recognition (OCR) tools to read the text from the detected number plates.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
