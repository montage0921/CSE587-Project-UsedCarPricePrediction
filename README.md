# CSE587-Project-UsedCarPricePrediction

This README.md only covers the source code for Phase I and Phase II. The source code for Phase III (Data Product) is located in the master branch. A separate README.md in the `master` branch provides detailed information about the folder structure and app building process.

Predicting Used Car Market Value Using Machine Learning with Data Collected from Online Trading Platforms and Public Datasets

Team Number: 2  
Team Member:  
● 50608573 Te Shi teshi@buffalo.edu (git_id:montage0921)  
● 50388599 Shijie Zhou shijiezh@buffalo.edu  
● 50602483 Jiabao Yao jyao27@buffalo.edu  
● 50606804 Chao Wu cwu64@buffalo.edu

# HIGHLIGHT
>Phase I/Data Scrape/Car_Info.py

contains a scraping script we designed and developed from scratch using Selectolax and Playwright to collect our dataset.

>Phase II/Model_Q1_2_Te_Shi/gbc_mileage.ipynb
>Phase II/Model_Q1_2_Te_Shi/generate_uncommon.ipynb

Use Gradient Boosting Classifier in conjunction with a customized sampling algorithm to predict mileage ranges of a car.

>/Phase II/Model_Q3_4_Jiabao_Yao/color_fuel_vs_price_Model(Q3_Q4).ipynb

Use SelectKBest and Random Forest to perform feature selection and CatBoostClassifier with GridSearchCV to analyze the relationship between fuel type and other features.

>Phase II/Model_Q5_6_Chao_Wu/brand_price_q5.py

Using an XGBoost model and an LSTM neural network to analyze the impact of brand on resale values.

>Phase II/Model_Q7_8_Shijie_Zhou/NeuralNetworkClassifier(Q7)p2.ipynb

Build a neural network classifier to predict accident probability.

>Master branch: pages/4-admin.py + Database_Operation

An interactive admin page with authentication that allows users to perform CRUD operations on an SQL database, built on AWS RDS, in a user-friendly way

>Master branch:homepage.py

A modern UI page that provides car price predictions built using Streamlit

# Phase I

> Report

CSE587 Phase I report.pdf contains all the work we did for Project Phase I, including outputs of each group member's EDA operations.

> Dataset

Dataset/carinfo_after_pre_clean.csv

> Data Scrape

Dataset is obtained by scraping carmax.com. The scrape script is Car_info.py in folder "Data Scrape". "Data_combine.py" is used to merge all the csv files generating during scraping process into one.

> Data Cleaning and Processing

"preliminary_data_processing_cleaning.ipynb" in "Data Cleaning" folder performed some general cleaning/processing for our original dataset.

> EDA

"EDA" folder contains each group memember's ipynb files for EDA operations
`~/Phase I/EDA`

# Phase II

> Report

CSE587 Phase II report.pdf contains all the work we did for Project Phase II, including questions refining in phase I, model training and visulizations of each group member.

> Model

`Model Path: ~/Phase II/`  
There are four corresponding folders to anaylysis and answer the questions proposed in phase I.

**Question 1:** What features could be affected by mileage? (Te Shi)  
`EDA path: ~/Phase II/Model_Q1_2_Te_Shi/Q1_EDA_Refined.ipynb`  
`Phase II Q1 path: ~/Phase II/Model_Q1_2_Te_Shi/gbc_mileage.ipynb`
`Phase II Customized Sample Generation Algorithm: ~/Phase II/Model_Q1_2_Te_Shi/generate_uncommon.ipynb`

**Question 2:** Do different car classes exhibit significant and distinct characteristics across features? (Te Shi)  
`EDA path: ~/Phase II/Model_Q1_2_Te_Shi/Q2_EDA_Refined.ipynb`  
`Model path: ~/Phase II/Model_Q1_2_Te_Shi/`
`Phase II Q2 path: ~/Phase II/Model_Q1_2_Te_Shi/decision_tree_class.ipynb`

**Dataset used for Q1 and Q2**
`Phase II Q2 path: ~/Phase II/Model_Q1_2_Te_Shi/processed_data_mileage.csv`

**Question 3:** Does color matter for used car prices and how does it affect them? (Jiabao Yao)  
`EDA path: ~/Phase I/EDA/color_fuel_vs_price(Q3_Q4).ipynb`  
`Model Path: ~/Phase II/Model_Q3_4_Jiabao_Yao/color_fuel_vs_price_Model(Q3_Q4).ipynb`

**Question 4:** What attributes are associated with fuel for used cars? (Jiabao Yao)  
`EDA path: ~/Phase II/Model_Q3_4_Jiabao_Yao/color_fuel_vs_price_EDA(Q4_refine).ipynb`  
`Model Path: ~/Phase II/Model_Q3_4_Jiabao_Yao/color_fuel_vs_price_Model(Q3_Q4).ipynb`

**Question 5:** Does the resale price of a particular brand get influenced by the resale prices of competing brands, and if so, is there a lag effect in this influence? (Chao Wu)
`EDA path: ~/Phase I/EDA/owner_brand_vs_price(Q5_Q6).ipynb`  
`Model Path&Analysis Path: ~/Phase II/Model_Q5_6_Chao_Wu/brand_price_q5.py`

**Question 6:** Is the number of owners a significant factor influencing resale price? Given the current imbalance in the distribution of owner counts, could data augmentation help create a more balanced distribution and enhance the predictive power of this feature? (Chao Wu)  
`EDA path: ~/Phase I/EDA/owner_brand_vs_price(Q5_Q6).ipynb`    
`Model Path&Analysis Path: ~/Phase II/Model_Q5_6_Chao_Wu/smote_owner_price_q6.py`

**Question 7:** How do the accidents or damage records of the used cars affect the resale price? (Shijie Zhou)  
`EDA path: ~/Phase I/EDA/damage_vs_price(Q7).ipynb `  
`Phase II Model&Analysis Path: ~/Phase II/Model_Q7_8_Shijie_Zhou/NeuralNetworkClassifier(Q7)p2.ipynb`

**Question 8:** For used cars with different makes, will the accident record affect the used cars’ price differently? (Shijie Zhou)  
`EDA path: ~/Phase I/EDA/damage_vs_price_make(Q8).ipynb `  
`Phase II Model&Analysis Path: ~/Phase II/Model_Q7_8_Shijie_Zhou/Gaussian_Mixture_Model(Q8)p2.ipynb`
