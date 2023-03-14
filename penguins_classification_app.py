# IMPORTING REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import time
import pickle # To load saved Machine Learning model,Scaler,One Hot Encoder
import streamlit as st
from streamlit_lottie import st_lottie # Used to import Lottie files
import streamlit_option_menu # Used for Navigation Bar
import requests
import json
from PIL import Image # Used to load images
from streamlit_extras.dataframe_explorer import dataframe_explorer # Used to display dataframe in more interactive format

# SETTING PAGE CONFIGURATION
st.set_page_config(page_title='Penguins Classification',layout='wide')
   
streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

			html, body, [class*="css"]  {
			font-family: 'sans-serif';
			}
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)
# LOADING THE SAVED DECISIONTREE CLASSIFIER MODEL,ONE-HOT ENCODER,SCALER
loaded_model = pickle.load(open('penguins_DTclassification.sav','rb'))
loaded_encoder = pickle.load(open('penguins_classification_ohe.sav','rb'))
loaded_scaler = pickle.load(open('penguins_classification_scaler.sav','rb'))

# CREATING A FUNCTION THAT MAKES PREDICTION USING LOADED MODEL
def penguin_classifier(data):
    input_data =np.hstack(([[data[1],data[2],data[3],data[4]]],loaded_encoder.transform([[data[0],data[5]]])))
    scaled_input_data = loaded_scaler.transform(input_data)
    prediction = loaded_model.predict(scaled_input_data)
    return prediction[0]

def main():          
# USING LOCAL CSS
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
    local_css('style.css')
    
# CREATING NAVIGATION BAR WITH OPTION_MENU    
selected = streamlit_option_menu.option_menu(menu_title=None,options=['Intro','Data','Make Prediction'],icons=['house','activity','book'],menu_icon='list',default_index=0,orientation='horizontal',styles={
            "container": {"padding": "0!important", "background-color": "#white"},
            "icon": {"color": "yellow", "font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "middle", "margin":"0px", "--hover-color": "grey"},
            "nav-link-selected": {"background-color": 'blue'}
        })
# CREATING HOME PAGE    
if selected=='Intro':
    st.title(':blue[Machine Learning]')
    
    # CONTAINER TO DISPLAY TEXT AND IMAGE ABOUT BRIEF INTRO OF MACHINE LEARNING
    with st.container():
        st.write('- - -')
        text_column,image_column = st.columns((2,1))
        with text_column:
            st.header(':blue[Brief Introduction about Machine Learning]')
            st.write('''Machine Learning can be defined with many definitions.One way to define it is "The field of study that gives computers the ability to learn without being explicitly programmed".Machine learning is programming computers to optimize a performance criterion using example data or past experience .We have a model defined up to some parameters, and learning is the execution of a computer program to optimize the parameters of the model using the training data or past experience.''')
            st.write("There are several types of machine learning, including **supervised learning**, **unsupervised learning**, and **reinforcement learning**.Supervised learning involves training a model on labeled data, while unsupervised learning involves training a model on unlabeled data. Reinforcement learning involves training a model through trial and error.Machine learning is used in a wide variety of applications, including image and speech recognition, natural language processing, and recommender systems.")
        with image_column:
            st.image(Image.open('ml_image.png.jpeg'),width=500) 
            
    # CONTAINER TO DISPLAY TEXT AND IMAGE ABOUT BRIEF INTRO OF REGRESSION           
    with st.container():
        st.write('- - -')
        st.header(':blue[Regression]')
        st.write('   ')
        image_column,text_column = st.columns(2)
        with image_column:
            st.image(Image.open('regression.png'),width=400)   
        with text_column:   
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write('''Machine Learning Regression is a technique for investigating the relationship between independent variables or features and a dependent variable or outcome. It is used as a method for predictive modelling in machine learning,in which an algorithm is used to predict continuous outcomes.Regression is a field of study in statistics which forms a key part of forecast models in machine learning.''')
            st.subheader(':orange[Linear Regression]')
            st.write('''As the name suggests, it assumes a linear relationship between a set of independent variables to that of the dependent variable.Figure shown left side,represents a case of linear regression,where we can observe some kind of linear relation between Y and X.''')
            st.subheader(':orange[Polynomial Regression]')
            st.write('''In polynomial regression, the relationship between the independent variable x and the dependent variable y is described as an nth degree polynomial in x.That n can be 1 or 2 or 3... If n=1 then it represents linear case.''')
    st.write(' ')
    st.write(' ')
    st.write(' ') 
           
    st.header(':blue[Types of Regression Algorithms]')
    # CONTAINER TO DISPLAY TYPES OF REGRESSION ALGORITHMS IMAGE
    with st.container(): 
        col1,col2,col3 = st.columns([1,6,1])
        with col1:
            st.write(' ')
        with col2:     
            st.write(' ')
            st.write(' ')  
            st.image(Image.open('types_regression.png.webp'),width=600) 
        with col3:
            st.write('  ') 
            
    # CONTAINER TO DISPLAY TEXT AND IMAGE ABOUT BRIEF INTRO OF CLASSIFICATION                                       
    with st.container():   
        st.write('- - -')
        st.header(':blue[Classification]')
        st.write('   ')
        image_column,text_column = st.columns(2)
        with image_column:
            st.image(Image.open('classification.png'))
        with text_column:
            st.write('''In Machine Learning, Classification is a predictive modeling problem where the class label is anticipated for a specific example of input data. For example, in determining handwriting characters, identifying spam, and so on, the classification requires training data with a large number of datasets of input and output.''')
            st.subheader(':orange[Binary classification]')
            st.write('''Binary is a type of problem in classification in machine learning that has only two possible outcomes. For example, yes or no, true or false, spam or not spam, etc. Some common binary classification algorithms are logistic regression, decision trees, simple bayes, and support vector machines.''')
            st.subheader(':orange[Multiclass classification]')
            st.write('''Multi-class is a type of classification problem with more than two outcomes and does not have the concept of normal and abnormal outcomes. Here each outcome is assigned to only one label. For example, classifying images, classifying species, and categorizing faces, among others. Some common multi-class algorithms are choice trees, progressive boosting, nearest k neighbors, and rough forest.''')
    st.write(' ')
    st.write(' ')
    st.write(' ')
      
    st.header(':blue[Classification Techniques]')
    # CONTAINER TO DISPLAY TYPES OF CLASSIFICATION ALGORITHMS IMAGE
    with st.container(): 
        col1,col2,col3 = st.columns([2,6,1])
        with col1:
            st.write('')
        with col2:       
            st.image(Image.open('types_Classification.png'),width=600) 
        with col3:
            st.write('')      
    st.write('- - -')

# CREATING A PAGE TO GIVE INFORMATION ABOUT OUR PROJECT
# LOADING DATASET WITH PANDAS
df = pd.read_csv('penguins_df.csv')

# CREATING PAGE FOR DATA DESCRIPTION
if selected=='Data':
    # TITLE
    st.title(':blue[Palmer Penguins Dataset]')
    #st.write(df.sample(250,ignore_index=True)
    st.write('''Data were collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.
    Gorman KB, Williams TD, Fraser WR (2014) Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus Pygoscelis).''')
    st.dataframe(dataframe_explorer(df),use_container_width=True)
    st.write('- - -')
    # DISPLAYING FEATURES/COLUMNS IN DATASET
    st.markdown("<p class='font'>Features of Data</p>",unsafe_allow_html=True)
    st.write(' ')
    st.markdown('- Species : penguin species (Chinstrap, Adelie, or Gentoo)')
    st.markdown('- Island :  island name (Dream, Torgersen, or Biscoe) in the Palmer Archipelago (Antarctica)')
    st.markdown('- Culmen Length (in mm)')
    st.markdown('- Culmen Depth (in mm)')
    st.markdown('- Flipper Length (in mm)')
    st.markdown('- Body Mass (in grams)')  
    st.markdown('- Sex')  
    st.write('- - -')
    
    # DISPLAYING IMAGES OF DIFFERENT SPECIES OF PENGUINS 
    st.markdown("<p class='font'>Species Images</p>",unsafe_allow_html=True)
    # CREATING CONTAINER WITH 3 COLUMNS FOR 3 DIFFERENT SPECIES OF PENGUINS
    with st.container():
        adelie_image,chinstrap_image,gentoo_image = st.columns((1,1,1))
        with adelie_image:
            st.write(' ')
            st.image(Image.open('adelie_image.png.jpeg'))
            st.subheader(':green[Adelie Penguin]')
        with chinstrap_image:
            st.write(' ')
            st.image(Image.open('chinstrap_image.png.jpeg')) 
            st.subheader(':green[Chinstrap Penguin]')
        with gentoo_image:
            st.write(' ')
            st.image(Image.open('gentoo_image.png.jpeg'),width=400)
            st.subheader(':green[Gentoo Penguin]')       
        st.write('- - -')
        
    # CREATING A CONTAINER TO DISPLAY IMAGES ABOUT CULMEN AND FLIPPER   
    with st.container():
        culmen_column,flipper_column = st.columns(2)
        with culmen_column:
            st.write(' ')
            st.image(Image.open('culmen_length_depth.png.jpeg'))
            st.subheader(':green[Culmen Length,Culmen_depth]')
            st.write("Culmen is the upper ridge of the Penguin's bill(beak).In above image,we can see how the culmen length and culmen depth")
        with flipper_column:
            st.write(' ')
            st.image(Image.open('flipper_len.png.jpeg'),width=320)
            st.subheader(':green[Flipper Length]')
            st.write("Flipper's are wings of penguins.In above image,we can see how the flipper length is measured")
    st.write('- - -')
    st.markdown("<p class='font'>Procedure Followed</p>",unsafe_allow_html=True)
    st.write('On taking features of Penguin that are Island,Culmen Length,Culmen Depth,Flipper Length,Body Mass,Sex of Penguin,,biologists had found their species.We have predefined data about features of penguin and their species.I build a Machine Learning Classification model using Decision Tree Algorithm.In Prediction page,you can give inputs and see the prediction.')        
    st.markdown('- I downloaded the dataset and loaded it in,by using Pandas.')
    st.markdown('- Using Python Libraries Numpy,Matplotlib and Seaborn,I did data exploration,made some interactive plots using Plotly.')
    st.markdown('- Importing required libraries from Scikit Learn Library.')
    st.markdown("- Splitting the dataset's features(X) and target(y) into training and testing data.")
    st.markdown('- Data Preprocessing that involves encoding the categorical features and scaling all features.')       
    st.markdown('- Creating instance of Classification algorithms and fitting with training data.')
    st.markdown('- Comapring metrics of classification models using Confusion Matrix')
    st.markdown('- Finally selecting best model with best parameters using GridSearchCV.And Saving that model.')
    st.write('You can see Python code for Machine Learning [Here](https://github.com/TRGanesh/penguins_classification1/blob/main/penguins_DTClassification.ipynb)')
    st.write('You can see Python code for Streamlit web page [Here](https://github.com/TRGanesh/penguins_classification1/blob/main/penguins_classification_app.py)')
st.markdown(''' <style> .font{font-size:30px;font-weight:bold;
            font-family:'Copper Black';
            color:#FF9633;}</style>''',unsafe_allow_html=True)
        

# CREATING A PAGE FOR MODEL PREDICTION    
if selected=='Make Prediction':   
    # TITLE
    st.title(':blue[Penguins Classification] :penguin:')
    
    
    # CONTAINER TO DISPLAY A FORM(TO TAKE INOUTS FROM USER) AND AN IMAGE(OF 3 PENGUINS)
    with st.container():
        st.write('- - -')
        left_column,right_column = st.columns((2,1)) 
        with left_column:
            # GETTING DATA FROM USER
            Island = st.selectbox("**Island**",['Biscoe','Dream','Torgersen'],key='island')	
            culmen_length_mm = st.text_input("**Culmen Length**",placeholder='Enter culmen length(in mm)')
            culmen_depth_mm	= st.text_input('**Culmen Depth**',placeholder='Enter culmen depth(in mm)')
            flipper_length_mm = st.text_input('**Flipper Length**',placeholder='Enter flipper length(in mm)')
            body_mass_g	= st.text_input('**Body Mass**',placeholder='Enter body mass(in grams)')
            sex = st.selectbox('**Sex**',['FEMALE','MALE'],key='sex')       
        with right_column:
            st.write(' ');st.write(' ');st.write(' ')
            st.write(' ');st.write(' ');st.write(' ')
            st.write(' ');st.write(' ');st.write(' ')
            st.image(Image.open('3_penguins.png.jpeg'))
        
        # DISPLAYING RESULT OUTSIDE OF CONTAINER
        st.markdown("<p class='font'>Model Prediction</p>",unsafe_allow_html=True)
        answer = ''
        
        # CREATING A BUTTON,ON CLICKING IT WE WILL BE ABLE TO SEE RESULT AS IMAGE
        if st.button('Result'):
            answer = penguin_classifier([Island,culmen_length_mm,culmen_depth_mm,flipper_length_mm,body_mass_g,sex])
            # st.write('Penguin is classified as')
            if answer == 'Gentoo':
                st.image(Image.open('gentoo_image.png.jpeg'),width=400)
            elif answer == 'Chinstrap':
                st.image(Image.open('chinstrap_image.png.jpeg'))    
            elif answer == 'Adelie':
                st.image(Image.open('adelie2_image.png.jpeg'))
            st.success(f'Penguin is classified as {answer}',icon="✅")
        
if __name__ == "__main__":
    main()
