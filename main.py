import os
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import numpy as np
import cv2
import joblib
import tensorflow as tf
from gtts import gTTS
from playsound import playsound
import pygame
import sklearn

from sklearn import set_config
set_config(assume_finite=True)

parkinsons_model = joblib.load('parkinsons.pkl')
heart_model = joblib.load('heart_disease_model.pkl')
kidney_model = joblib.load('kidney_disease.pkl')
cancer_model = joblib.load('cancer_data.pkl')
Diabities_model =joblib.load('diabetes.pkl') 
liver_model = joblib.load('liver_data.pkl')
malaria_model = tf.keras.models.load_model('malaria_cells.h5')
covid_model = tf.keras.models.load_model('Covid.h5')

selected = None

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Sidebar for page selection
st.sidebar.title("Health Assistance")
page = st.sidebar.selectbox("Choose a page:", ["Info Page", "Multiple Disease Prediction System"])


image_paths= {'Diabetes':'D:\\Desktop\\Projects\\New Mutiple Disease project\\diabities.jpg',
              'Heart':'D:\\Desktop\\Projects\\New Mutiple Disease project\\heart.jpg',
              'Parkinsons':'D:\\Desktop\\Projects\\New Mutiple Disease project\\parkisons.jpg',
              'Cancer':'D:\\Desktop\\Projects\\New Mutiple Disease project\\cancer image.jpg',
              'Liver':'D:\\Desktop\\Projects\\New Mutiple Disease project\\liver.jpg',
              'Kidney':'D:\\Desktop\\Projects\\New Mutiple Disease project\\kidney.png'}

images = {}
for disease, path in image_paths.items():
    img = Image.open(path)
    images[disease] = img



# Show content based on selection
if page == "Info Page":
    # App title
    st.title("Welcome to My Streamlit App of Multiple Disease Prediction System")

    # Text to convert to speech
    title = "Welcome to My Streamlit App of Multiple Disease Prediction System"

    # Output file path
    output_file = "output/output.mp3"

    # Ensure the directory exists
    os.makedirs("output", exist_ok=True)

    # Generate and save the speech if not already done
    if not os.path.exists(output_file):  # Avoid regenerating the file
        tts = gTTS(text=title, lang='en')
        tts.save(output_file)

    # Automatically play the audio using playsound
    playsound(output_file)

    # Streamlit audio component to play the audio
    st.audio(output_file, format='audio/mp3')
    st.image('D:\\Desktop\\Projects\\New Mutiple Disease project\\diabities.jpg', width=600)
    st.write("""
    This is an informational page where I introduce about project of Multiple Disease Prediction app.
    The project "Multiple Disease Prediction using Machine Learning and Streamlit" focuses on predicting five different diseases: 
    
        - Cancer Disease
        - Diabetes Diseases
        - Heart Disease
        - Parkinson's Disease
        - Kidney Disease
        - Liver Disease
        - Malaria Disease
        - Covid Disease

     The application is deployed using Streamlit.""")


if page=="Multiple Disease Prediction System":

 with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction','Cancer Prediction',
                            'Liver Disease Prediction','Kidney Disease Prediction','Malaria Disease Prediction',
                            'Covid Disease Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'activity',
                                  'person','heart','activity','heart'],
                           default_index=0)



# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')
    st.image(images['Diabetes'], width=600)

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0 ,value=1)

    with col2:
        Glucose = st.number_input('Glucose Level', min_value=40.0 ,value=120.0)

    with col3:
        BloodPressure = st.number_input('Blood Pressure value',min_value=20.0 ,value=80.0)

    with col1:
        SkinThickness = st.number_input('Skin Thickness value',min_value=5.0 ,value=20.0)

    with col2:
        Insulin = st.number_input('Insulin Level',min_value=10.0 ,value=50.0)

    with col3:
        BMI = st.number_input('BMI value',min_value=17.0 ,value=25.0)

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value',min_value=0 ,value=1)

    with col2:
        Age = st.number_input('Age of the Person',min_value=3 ,value=30)

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = Diabities_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')
    st.image(images['Heart'], width=600)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age',min_value=30.0 ,value=80.0)

    with col2:
        cp = st.number_input('Chest Pain types',min_value=0.0 ,value=3.0)

    with col3:
        trestbps = st.number_input('Resting Blood Pressure',min_value=110.0 ,value=150.0)

    with col1:
        chol = st.number_input('Serum Cholestoral in mg/dl',min_value=200.0 ,value=360.0)

    with col2:
        restecg = st.number_input('Resting Electrocardiographic results',min_value=0.0 ,value=1.0)

    with col3:
        thalach = st.number_input('Maximum Heart Rate achieved',min_value=110.0 ,value=190.0)

    with col1:
        exang = st.number_input('Exercise Induced Angina',min_value=0.0 ,value=1.0)

    with col2:
        oldpeak = st.number_input('ST depression induced by exercise',min_value=0.0 ,value=3.8)

    with col3:
        slope = st.number_input('Slope of the peak exercise ST segment',min_value=0.0 ,value=2.0)

    with col1:
        ca = st.number_input('Major vessels colored by flourosopy',min_value=0.0 ,value=0.0)

    with col2:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect',min_value=1.0 ,value=3.0)

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age,cp, trestbps, chol,restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")
    st.image(images['Parkinsons'], width=600)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)',min_value=90.0 ,value=180.0)

    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)',min_value=110.0 ,value=350.0)

    with col3:
        flo = st.number_input('MDVP:Flo(Hz)',min_value=70.0 ,value=150.0)

    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)',min_value=0.00001 ,value=0.000019)

    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)',min_value=0.00001 ,value=0.000019)

    with col1:
        RAP = st.number_input('MDVP:RAP',min_value=0.00001 ,value=0.00790)

    with col2:
        PPQ = st.number_input('MDVP:PPQ',min_value=0.00001 ,value=0.00790)

    with col3:
        DDP = st.number_input('Jitter:DDP',min_value=0.00001 ,value=0.00790)

    with col4:
        Shimmer = st.number_input('MDVP:Shimmer',min_value=0.00001 ,value=0.00790)

    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)',min_value=0.00001 ,value=0.00790)

    with col1:
        APQ3 = st.number_input('Shimmer:APQ3',min_value=0.1 ,value=0.700)

    with col2:
        APQ5 = st.number_input('Shimmer:APQ5',min_value=0.0350 ,value=0.09403)

    with col3:
        APQ = st.number_input('MDVP:APQ',min_value=0.0350 ,value=0.09403)

    with col4:
        DDA = st.number_input('Shimmer:DDA',min_value=0.0350 ,value=0.09403)

    with col5:
        NHR = st.number_input('NHR',min_value=0.0107 ,value=0.02919)

    with col1:
        HNR = st.number_input('HNR',min_value=15.00 ,value=30.00)

    with col2:
        RPDE = st.number_input('RPDE',min_value=0.4406 ,value=0.458359)

    with col3:
        DFA = st.number_input('DFA',min_value=0.68608 ,value=0.825288)

    with col4:
        spread1 = st.number_input('spread1',min_value=-6.1051 ,value=-5.18692)

    with col5:
        spread2 = st.number_input('spread2',min_value=0.30559 ,value=0.311173)

    with col1:
        D2 = st.number_input('D2',min_value=1.00000 ,value=3.007463)

    with col2:
        PPE = st.number_input('PPE',min_value=0.1001 ,value=0.119308)

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
# Cancer Prediction

if selected == 'Cancer Prediction':

    # page title
    st.title('Cancer Prediction using ML')
    st.image(images['Cancer'], width=600 )

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Id = st.number_input('Id',min_value=0.00001 ,value=0.00790)

    with col2:
        area_mean = st.number_input('area_mean',min_value=0.00001 ,value=0.00790)

    with col3:
         Radius_mean= st.number_input('Radius_mean',min_value=1.0 ,value=25.0)

    with col1:
        Texture_mean = st.number_input('texture_mean',min_value=9.0 ,value=40.0)

    with col2:
        Perimeter_mean = st.number_input('perimeter_mean',min_value=100.0 ,value=150.0)

    with col3:
        smoothness_mean = st.number_input('smoothness_mean',min_value=0.0700 ,value=0.08129)

    with col1:
        compactness_mean = st.number_input('compactness_mean',min_value=0.01938 ,value=0.34540)

    with col2:
        concavity_mean = st.number_input('concavity_mean',min_value=0.0000 ,value=0.4268)

    with col3:
        concave_points_mean=st.number_input('concave points_mean',min_value=0.0000 ,value=0.2012)

    diagnosis = st.selectbox('diagnosis', options=['B', 'M'])

    # creating a button for Prediction

    if st.button('Cancer Test Result'):

        user_input = np.array([[Id,Radius_mean,Texture_mean,Perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean]])
        cancer_prediction = cancer_model.predict(user_input)
        result = 'The person is caused by cancer' if cancer_prediction[0] == 1 else 'The person is not caused by cancer'
        st.success(result)

# Liver Prediction Page
if selected == 'Liver Disease Prediction':

    # page title
    st.title('Liver Disease Prediction using ML')
    st.image(images['Liver'], width=600)

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input('Age',min_value=4.0 ,value=90.0)

    with col2:
        Gender = st.number_input('Gender',min_value=0.0 ,value=1.0)

    with col3:
        Total_Bilirubin = st.number_input('Total_Bilirubin',min_value=0.4 ,value=75.0)

    with col1:
        Direct_Bilirubin = st.number_input('Direct_Bilirubin',min_value=0.1,value=19.7)

    with col2:
        Alkaline_Phosphotase = st.number_input('Alkaline_Phosphotase',min_value=63.0 ,value=2110.0)

    with col3:
        Alamine_Aminotransferase = st.number_input('Alamine_Aminotransferase',min_value=10.0 ,value=2000.0)

    with col1:
        Aspartate_Aminotransferase = st.number_input('Aspartate_Aminotransferase',min_value=2.7 ,value=9.6)

    with col2:
        Total_Protiens = st.number_input('Total_Protiens',min_value=0.9 ,value=5.5)

    with col3:
        Albumin=st.number_input('Albumin',min_value=0.3 ,value=2.8)

    with col1:
        Albumin_and_Globulin_Ratio=st.number_input('Albumin_and_Globulin_Ratio',min_value=1.0,value=2.0)

    gender = st.selectbox('Gender', options=['M', 'F'])

    if st.button('Liver Test Result'):
        user_input = [Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]
        user_input = [float(x) for x in user_input]
        lvr_liver_prediction = liver_model.predict([user_input])
        result = 'The person is caused by Liver disease' if lvr_liver_prediction[0] == 1 else 'The person is not caused by Liver disease'
        st.success(result)

# Kidney Prediction Page
if selected == 'Kidney Disease Prediction':

    # page title
    st.title('Kidney Disease Prediction using ML')
    st.image(images['Kidney'], width=600)

    # getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Age = st.number_input('age',min_value=0.0000 ,value=75.00)

    with col2:
        Bp = st.number_input('Bp',min_value=50.0 ,value=80.0)

    with col3:
        Sg = st.number_input('Sg',min_value=1.0 ,value=1.02)

    with col4:
        Al = st.number_input('Al',min_value=0.0 ,value=0.1)

    with col1:
        Su = st.number_input('Su',min_value=1.0 ,value=10.0)

    with col2:
        Rbc = st.number_input('Rbc',min_value=0.0 ,value=2.0)

    with col3:
        Pc = st.number_input('Pc',min_value=0.0 ,value=2.0)

    with col4:
        Cad = st.number_input('Cad',min_value=1.0 ,value=5.0)

    with col1:
        Appet = st.number_input('Appet',min_value=1.0 ,value=2.0)

    rbc = st.selectbox('Rbc', options=['10', 'Normal', 'Abnormal'])
    pc = st.selectbox('Pc', options=['Normal', 'Abnormal'])
    cad = st.selectbox('Cad', options=['Yes', 'No'])
    appet = st.selectbox('Appet', options=['Good', 'Poor'])

    kid_kidney = ''

    # creating a button for Prediction

    if st.button('Kidney Test Result'):

        user_input = [Age, Bp, Sg, Al, Su, Rbc, Pc,
                      Cad, Appet]
        user_input = [float(x) for x in user_input]

        kid_kidney_prediction = kidney_model.predict([user_input])

        result = 'The person is caused by Kidney disease' if kid_kidney_prediction[0] == 1 else 'The person is not caused by Kidney disease'
        st.success(result)

if selected == 'Malaria Disease Prediction':
    st.title("Malaria Disease Prediction")

    labels = ['Parasitized', 'Uninfected']

    # User input for uploading an image
    uploaded_image = st.file_uploader("Upload a Cell Image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Convert the uploaded image into a format usable by OpenCV
        image = Image.open(uploaded_image)
        image = image.convert('RGB')
        image = np.array(image)

        # Resize the image to the input size expected by the model (e.g., 128x128 for CNN)
        image_resized = cv2.resize(image, (128, 128))

        # Normalize pixel values (assuming the model expects this)
        image_scaled = image_resized / 255.0

        # Reshape image to match model input (e.g., for CNN model)
        image_reshaped = np.reshape(image_scaled, [1, 128, 128, 3])

        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=False)

        # Button to trigger the malaria test result
        if st.button('Malaria Test Result'):
            try:
                # Ensure `malaria_model` is the loaded Keras model
                if isinstance(malaria_model, tf.keras.Model):
                    # Predict using the model
                    pred = malaria_model.predict(image_reshaped)

                    # Check the shape of the prediction output
                    if pred.shape[-1] == 1:
                        # Single probability output (e.g., sigmoid activation)
                        parasitized_prob = pred[0][0]
                        uninfected_prob = 1 - parasitized_prob
                    elif pred.shape[-1] == 2:
                        # Two-class probability output (e.g., softmax activation)
                        parasitized_prob = pred[0][0]
                        uninfected_prob = pred[0][1]
                    else:
                        raise ValueError("Unexpected model output shape.")

                    # Determine the result based on the highest probability
                    result = labels[np.argmax([parasitized_prob, uninfected_prob])]

                    # Display the final classification
                    st.success(f"The uploaded image is classified as: {result}")
                else:
                    st.error("Model is not loaded correctly.")
            except Exception as e:
                # Catch any errors during prediction and show the error message
                st.error(f"An error occurred: {str(e)}")


if selected == 'Covid Disease Prediction':
    st.title("Covid Disease Prediction")

    labels = ['Covid', 'Non_Covid']

    # User input for uploading an image
    uploaded_image = st.file_uploader("Upload a Cell Image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Convert the uploaded image into a format usable by OpenCV
        image = Image.open(uploaded_image)
        image = image.convert('RGB')
        image = np.array(image)

        # Resize the image to the input size expected by the model (e.g., 128x128 for CNN)
        image_resized = cv2.resize(image, (128, 128))

        # Normalize pixel values (assuming the model expects this)
        image_scaled = image_resized / 255.0

        # Reshape image to match model input (e.g., for CNN model)
        image_reshaped = np.reshape(image_scaled, [1, 128, 128, 3])

        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=False)

        # Button to trigger the malaria test result
        if st.button('Covid Test Result'):
            try:
                # Ensure `malaria_model` is the loaded Keras model
                if isinstance(covid_model, tf.keras.Model):
                    # Predict using the model
                    pred = covid_model.predict(image_reshaped)

                    # Check the shape of the prediction output
                    if pred.shape[-1] == 1:
                        # Single probability output (e.g., sigmoid activation)
                        covid_prob = pred[0][0]
                        non_covid_prob = 1 - covid_prob
                    elif pred.shape[-1] == 2:
                        # Two-class probability output (e.g., softmax activation)
                        covid_prob = pred[0][0]
                        non_covid_prob = pred[0][1]
                    else:
                        raise ValueError("Unexpected model output shape.")

                    # Determine the result based on the highest probability
                    result = labels[np.argmax([covid_prob, non_covid_prob])]

                    # Display the final classification
                    st.success(f"The uploaded image is classified as: {result}")
                else:
                    st.error("Model is not loaded correctly.")
            except Exception as e:
                # Catch any errors during prediction and show the error message
                st.error(f"An error occurred: {str(e)}")
