import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from PIL import Image
import base64
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import pickle
#from sklearn.svm import SVC, LinearSVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


st.set_page_config(
    page_title="Penguin App",
    page_icon='./penguin (1).png', layout="wide",
)


def footer(p):
    style = """
    <style>
        footer {visibility: hidden;}
        .stApp { bottom: 40px; }
         # prof_link {text-decoration: none;}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    st.markdown('''<p style='position:fixed; bottom:0; height:20px; text-align:centre; left:10px; right:0; font-size: 16px;'> 📧 Connect With <a id="prof_link" href="mailto:shivharevishal12@gmail.com"> Vishal Shivhare</a></p>''', unsafe_allow_html=True)


# top image
# img1 = Image.open('./person_fishing_with_penguine.jpg')
# img = img.resize((1000, 250))
# st.image(img1)
footer('p')
st.write("""<h1 align="center">🐧 Penguin Prediction App </h1>
<h5 align="center" >This app predicts the <b>Palmer Penguin</b> species! </h5>
<br>
""", unsafe_allow_html=True)

# <p align="center" >Data obtained from the <a href ="https://github.com/allisonhorst/palmerpenguins">palmerpenguins library</a> in R by Allison Horst.</p>

content, vectors = st.columns([1, 1], gap="large")
with content:
    st.header('Choose **ML** Model')
    model = st.selectbox('Model', ('Random Forest Classifier',
                                   'Logistic Regression', 'Support Vector Classifier', 'KNeighbors Classifier', 'Gaussian Naive Bayes', 'Decision Tree Classifier'))
with vectors:
    # st.image(img)
    # img1 = Image.open('./header_pengu.png')
    # img1 = img1.resize((100, 100))
    # st.image(img1)
    anim = load_lottiefile('hello_pengu.json')
    st_lottie(anim)
# side
# bar animation
# sidebar_pengu = load_lottiefile('sidebar_pengu.json')
# with st.sidebar:
#    st_lottie(sidebar_pengu)

img = Image.open('./new_body.jpg')
st.sidebar.image(img)

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader(
    "Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox(
            'Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider(
            'Bill length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider(
            'Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider(
            'Body mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0, ignore_index=True)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering

# save sex and island in varibale

pengu_island = df.at[0, 'island']
pengu_sex = df.at[0, 'sex']

encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]  # Selects only the first row (the user input data)

# Displays the user input features
st.write('''<br>''', unsafe_allow_html=True)
st.header('User Input Penguin Features')

if uploaded_file is not None:
    st.write(df)
    st.write("Bill Length (in mm) :-", df[0])
else:
    st.write(
        'Input Parameters can also be uploaded as CSV file (Like shown below) 👇🏻 ')
    st.write(df)
    x = df.iloc[:, 0]

features, vect1 = st.columns([1, 1], gap="large")
with features:
    st.write('''<br>''', unsafe_allow_html=True)
    st.header("Penguin Features")
    st.write("**Island :**", pengu_island)
    st.write("**Sex :**", pengu_sex)
    st.write("**Bill Length (in mm) :**", df.at[0, 'bill_length_mm'])
    st.write("**Bill Depth (in mm) :**", df.at[0, 'bill_depth_mm'])
    st.write("**Flipper length (in mm) :**", df.at[0, 'flipper_length_mm'])
    st.write("**Body Mass (in gram) :**", df.at[0, 'body_mass_g'])
with vect1:
    img = Image.open('./penguine_vetc1.png')
    # img = img.resize((400, 400))
    st.image(img)


# Reads in saved classification model
model_dict = {'Random Forest Classifier': './models/penguins_rf_clf.pkl',
              'Logistic Regression': './models/penguins_logreg_clf.pkl', 'Support Vector Classifier': './models/penguins_svc_clf.pkl', 'KNeighbors Classifier': './models/penguins_knn_clf.pkl',
              'Gaussian Naive Bayes': './models/penguins_gnb_clf.pkl', 'Decision Tree Classifier': './models/penguins_dt_clf.pkl'}


load_clf = pickle.load(open(model_dict[model], 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)
print(prediction)
print(prediction_proba)
#st.write(model_dict[model])
st.header('Predicted Species')
penguins_species = {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}
prediction = prediction[0]
st.write('''<br>''', unsafe_allow_html=True)
st.sucess("**The Predcited Species for Penguine Considering Input Features is  : "+ str(penguins_species[prediction]) + "**")

st.header('Prediction Probability')
#st.write(prediction_proba)
st.write("<h4>The Probability of Penguin Being of  Adelie Species is " + str(prediction_proba[0][0]) + "</h4>" ,unsafe_allow_html=True)
st.markdown('''The Probability of Penguin Being of  Chinstrap Species is''' + str(prediction_proba[0][1])+ ''' ''',unsafe_allow_html=True)
st.write("The Probability of Penguin Being of  Gentoo Species is",prediction_proba[0][2])


