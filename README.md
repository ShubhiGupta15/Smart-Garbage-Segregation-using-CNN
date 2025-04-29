# ♻️ Smart Garbage Segregation using CNN

An automated waste classification system that uses Convolutional Neural Networks (CNNs) to classify garbage into six categories: **plastic, paper, cardboard, metal, glass, and trash**. The system is deployed using Streamlit, allowing users to upload images and receive real-time classification predictions.

---

## 🌍 Project Motivation

Waste management is a global concern, and traditional manual sorting of garbage is inefficient and error-prone. Mismanaged waste leads to environmental pollution and ineffective recycling. This project aims to solve this issue by leveraging deep learning techniques — specifically CNNs — to **automate and improve the accuracy of waste segregation**.

---

## 🧠 Technologies & Concepts Used

- **Convolutional Neural Networks (CNNs)**
- **Image Classification**
- **Transfer Learning (planned for future improvements)**
- **Data Augmentation**
- **Hyperparameter Tuning**
- **Model Evaluation (Accuracy, Precision, Recall, F1-Score)**
- **Streamlit for Web App Deployment**
- **Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib**

---

## 📂 Dataset

- 📌 Source: [Kaggle - Garbage Classification using CNN](https://www.kaggle.com/code/tohidyousefi/garbage-classification-using-cnn/input)
- 🔢 Total images: 2,527 RGB images
- 📁 Categories: `plastic`, `paper`, `cardboard`, `metal`, `glass`, `trash`
- 📏 Images resized to **224x224**
- 📊 Data split: 80% training, 10% validation, 10% testing

---

## 🛠️ Project Workflow

### 1. **Data Preprocessing**
- Resized all images to 224x224 pixels
- Normalized pixel values to the range [0, 1]
- Data augmentation techniques:
  - Horizontal & vertical flipping
  - Shear transformations
  - Zooming
  - Random width and height shifts

### 2. **CNN Model Architecture**
- Multiple convolutional and pooling layers for feature extraction
- Dropout and batch normalization to reduce overfitting
- Fully connected layers for classification
- Activation function: ReLU & Softmax
- Loss function: Categorical Cross-Entropy
- Optimizer: Adam

### 3. **Model Training & Optimization**
- Trained for 100 epochs
- Batch size: Tuned through experimentation
- Learning rate: Tuned for best performance
- Metrics monitored: Accuracy, Precision, Recall, Loss

### 4. **Model Evaluation**
- Test Accuracy: **71.87%**
- Precision: **70.98%**
- Recall: **64.68%**
- F1-Score: **71.2%**
- Confusion Matrix: Visualized misclassifications across classes

### 5. **Model Deployment**
- Trained model saved in `.h5` format
- Deployed using **Streamlit** to build an interactive web app
- Upload images via:
  - File uploader
  - Image URL link

<img width="412" alt="image" src="https://github.com/user-attachments/assets/1fa0080b-dfea-4859-a15e-da85b78b7803" />

---

## 🖼️ Sample Outputs

### 📌 Correct Classification
- **Category**: Metal
- **Confidence**: 93%

<img width="141" alt="image" src="https://github.com/user-attachments/assets/a8a532ff-a3c3-44f7-bab9-32ec6c511660" />


### 📉 Confusion Matrix Insight
- **Best Performance**: Paper (46 correctly classified samples)
- **Needs Improvement**: Underrepresented categories like trash

<img width="226" alt="image" src="https://github.com/user-attachments/assets/9f684c9a-5687-4d61-af1d-908c06d4cc1d" />


## 🔍 Challenges Faced
- **Misclassification between visually similar classes** (e.g., plastic vs. paper, glass vs. plastic).
- **Class imbalance**: The trash category had fewer samples, leading to variations in recall and precision.
- **Variations due to external lighting and image quality**: Some images were misclassified due to inconsistent lighting conditions.

## 🚀 Future Improvements
- **Increase Dataset Size**: A larger dataset can improve generalization and performance, especially for underrepresented categories.
- **Apply Transfer Learning**: Using pre-trained models such as ResNet, VGG, or EfficientNet can enhance the model's performance on more complex image features.
- **Class Weighting/Oversampling**: Implement techniques to address class imbalance, especially for categories with fewer samples (e.g., trash).
- **GAN-based Data Augmentation**: Using Generative Adversarial Networks (GANs) for synthetic image generation can enhance model performance.
- **Optimize for Edge Devices**: Implement optimizations to run the model on small devices like Raspberry Pi for real-time waste classification.

## 🔑 Key Insights from Confusion Matrix
- The model performed well for certain categories like **paper**, with a high number of correct predictions.
- **Trash** and other underrepresented categories need further tuning due to the class imbalance.
- Misclassifications were observed between visually similar materials (e.g., plastic vs. glass).

## 📈 Experiments Conducted

- **Dataset**: The dataset consists of 2,527 labeled images of waste items in six categories.
- **Model**: Convolutional Neural Network (CNN) was used for classification.
- **Training**: The model was trained for 100 epochs with performance metrics monitored throughout the training process.

## 📝 Conclusion
This project demonstrates the application of CNNs in waste segregation, highlighting both the potential and challenges of using machine learning for real-world environmental issues. While the current model's accuracy is moderate, it lays a strong foundation for further improvements, including transfer learning, data augmentation, and model optimization for edge devices. With continuous development, this system can contribute to improving recycling efficiency and sustainability efforts in smart cities.

---

## 💻 How to Run the Project Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/smart-garbage-segregation.git
cd smart-garbage-segregation
