import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define CNN data preparation function
def CNN_data_preparation():
    # Assume dir_path and target_size are defined somewhere in the app
    dir_path = "/Users/shubhigupta/Desktop/Garbage Segregation/Smart-Garbage-Segregation-main/Dataset" # Set to your actual path
    target_size = (224, 224)  # Set to your desired target size for images
    
    train = ImageDataGenerator(horizontal_flip=True,
                               vertical_flip=True,
                               validation_split=0.1,
                               rescale=1./255,
                               shear_range=0.1,
                               zoom_range=0.1,
                               width_shift_range=0.1,
                               height_shift_range=0.1)
    
    test = ImageDataGenerator(rescale=1/255, validation_split=0.1)
    
    train_generator = train.flow_from_directory(directory=dir_path,
                                                target_size=target_size,
                                                class_mode="categorical",
                                                subset="training")
    
    test_generator = test.flow_from_directory(directory=dir_path,
                                              target_size=target_size,
                                              batch_size=251,
                                              class_mode="categorical",
                                              subset="validation")
    
    return train_generator, test_generator

# Preprocess function
def preprocess(image):
    image = image.convert('RGB')
    image = np.array(image.resize((224,224), Image.Resampling.LANCZOS))
    image = np.array(image, dtype='uint8')
    image = np.array(image)/255.0
    return image

# Generate labels function using train_generator
def gen_labels():
    train_generator, _ = CNN_data_preparation()
    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    return labels

# Load model and labels
labels = gen_labels()
model_path = '/Users/shubhigupta/Desktop/Garbage Segregation/Smart-Garbage-Segregation-main/mymodel.h5'
model = load_model(model_path, compile=False)

# Streamlit UI
html_temp = '''
  <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: -50px">
    <div style="display: flex; flex-direction: row; align-items: center; justify-content: center;">
     <center><h1 style="color: #000; font-size: 50px;"><span style="color: #0e7d73">Smart </span>Garbage</h1></center>
    </div>
    <div style="margin-top: -20px">
    <img src="https://i.postimg.cc/W3Lx45QB/Waste-management-pana.png" style="width: 400px;">
    </div>  
  </div>
'''
st.markdown(html_temp, unsafe_allow_html=True)

html_temp = '''
  <div>
    <center><h3 style="color: #008080; margin-top: -20px">Check the type here</h3></center>
  </div>
'''
st.markdown(html_temp, unsafe_allow_html=True)

# Image upload options
st.set_option('deprecation.showfileUploaderEncoding', False)
opt = st.selectbox("How do you want to upload the image for classification?", ('Please Select', 'Upload image via link', 'Upload image from device'))

image = None

# Image upload handling
if opt == 'Upload image from device':
    file = st.file_uploader('Select an image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        image = Image.open(file)

elif opt == 'Upload image via link':
    img_url = st.text_input('Enter the Image URL')
    if img_url:
        try:
            image = Image.open(urllib.request.urlopen(img_url))
        except Exception as e:
            st.error(f"Error loading image: {e}")

# Display uploaded image
if image is not None:
    st.image(image, width=224, caption='Uploaded Image')
    
    if st.button('Predict'):
        try:
            # Preprocess image before making prediction
            img = preprocess(image)
            
            # Make prediction
            prediction = model.predict(img[np.newaxis, ...])
            predicted_label = labels[np.argmax(prediction[0], axis=-1)]
            
            # Display prediction result
            st.success(f'Hey! The uploaded image has been classified as "{predicted_label}".')
        except Exception as e:
            st.error(f"Error during prediction: {e}")
