# Penguin-App

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)  
[![Python 3.8|3.9](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-360/)
![Type](https://img.shields.io/badge/Machine-Learning-red.svg) ![Type](https://img.shields.io/badge/Type-Supervised-yellow.svg)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://penguin-species-prediction.streamlit.app/)

![penguine_vetc1](https://user-images.githubusercontent.com/63242162/195385866-9fe36b93-a1bc-4967-9db6-c9bf2bd2f91f.png)

<br><br>
Penguin Species Prediction App is an end-to-end real-time web app which is designed to help users predict the species of penguins based on their characteristics such as bill length, bill depth, flipper length, body mass and sex in real time. <br>
Penguin Species Prediction App uses six different machine learning models to predict the species of penguins with high accuracy. The project is build on training multi-class ML models such as Random Forest Classifier, Logistic Regression, Support Vector Classifier, KNeighbors Classifier, Gaussian Naive Bayes and Decision Tree Classifier. GridCV is used for hyperparameter tuning for each model. <br>
The models are trained using the Palmer Archipelago (Antarctica) penguin data which contains data on three different species of penguins: Chinstrap, Adélie and Gentoo.
<br><br>
The web app allows users to choose the machine learning model they want to use for prediction. <br>
 It has 6 user input features and you can upload the input in CSV format or input them manually.<br>
The interactive input features allow you to get real-time species prediction by model of your choice and real-time probability prediction of each penguin species such as Chinstrap, Adélie and Gentoo.<br>


## Dataset

[Dataset](https://github.com/allisonhorst/palmerpenguins) Used were collected and made available by [Dr. Kristen
Gorman](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php)
and the [Palmer Station, Antarctica
LTER](https://pallter.marine.rutgers.edu/).
<br>
Datasets contain data for 344 penguins. There are 3 different 
species of penguins in this dataset, collected from 3 islands in the Palmer Archipelago, Antarctica.<br><br>
The dataset consists of 7 columns.

- species: penguin species (Chinstrap, Adélie, or Gentoo)
- culmen_length_mm: culmen length (mm)
- culmen_depth_mm: culmen depth (mm)
- flipper_length_mm: flipper length (mm)
- body_mass_g: body mass (g)
- island: island name (Dream, Torgersen, or Biscoe) in the Palmer Archipelago (Antarctica)
- sex: penguin sex


## Palmer penguins
<img src="https://raw.githubusercontent.com/vishulearnere/Penguin-App/main/penguin_specie.png" width="75%" style="display: block; margin: auto;" />

## Features

- Built and Trained **Multi Class ML Models** like
  - Random Forest Classifier
  - Logistic Regression
  - Support Vector Classifier
  - KNeighbors Classifier
  - Gaussian Naive Bayes
  - Decision Tree Classifier
  
- Used GridCV for hyperparameter Tuning for each model
- Bulit an End to End Real Time Web App for [Penguine Specie Prediction](https://penguin-species-prediction.streamlit.app/) 
- Deployed the Model On Streamlit Sharing 

## Web App Features

- Choose The Model which you want to apply from following 
  - Random Forest Classifier
  - Logistic Regression
  - Support Vector Classifier
  - KNeighbors Classifier
  - Gaussian Naive Bayes
  - Decision Tree Classifier
  
- 6 User Input Features 
- Upload The Input in CSV Format OR
- Input Them Manually 
- Interactive Input Features 
- Get Real Time Species Prediction By Model of Your Choice
- Get Real Time Probability Prediction oF Each Penguin Species 
  - Chinstrap
  - Adélie 
  - Gentoo
  
## Penguin-App Screencast


https://user-images.githubusercontent.com/63242162/195424157-0c2ef1f0-9087-4590-ad3c-c9af5ecd4d7f.mp4

