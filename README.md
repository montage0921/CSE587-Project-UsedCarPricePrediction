# CSE587-Project-UsedCarPricePrediction

Predicting Used Car Market Value Using Machine Learning with Data Collected from Online Trading Platforms and Public Datasets

Please put your team's information here. delete this message after you have done so.

Team Number: 2  
Team Member:  
● 50608573 Te Shi teshi@buffalo.edu (git_id:montage0921)  
● 50388599 Shijie Zhou shijiezh@buffalo.edu  
● 50602483 Jiabao Yao jyao27@buffalo.edu  
● 50606804 Chao Wu cwu64@buffalo.edu

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
`Model path: ~/Phase II/Model_Q1_2_Te_Shi/`

**Question 2:** Do different car classes exhibit significant and distinct characteristics across features? (Te Shi)  
`EDA path: ~/Phase II/Model_Q1_2_Te_Shi/Q2_EDA_Refined.ipynb`  
`Model path: ~/Phase II/Model_Q1_2_Te_Shi/`

**Question 3:** Does color matter for used car prices and how does it affect them?  (Jiabao Yao)  
`EDA path: ~/Phase I/EDA/color_fuel_vs_price(Q3_Q4).ipynb`  
`Model Path: ~/Phase II/Model_Q3_4_Jiabao_Yao`

**Question 4:** What attributes are associated with fuel for used cars? (Jiabao Yao)  
`EDA path: ～/Phase II/Model_Q3_4_Jiabao_Yao/color_fuel_vs_price_EDA(Q4_refine).ipynb`  
`Model Path: ~/Phase II/Model_Q3_4_Jiabao_Yao`

**Question 5:** Does the resale price of a particular brand get influenced by the resale prices of competing brands, and if so, is there a lag effect in this influence? (Chao Wu)

**Question 6:** Is the number of owners a significant factor influencing resale price? Given the current imbalance in the distribution of owner counts, could data augmentation help create a more balanced distribution and enhance the predictive power of this feature? (Chao Wu)

**Question 7:** How do the accidents or damage records of the used cars affect the resale price? (Shijie Zhou)
`EDA path: ~/Phase I/EDA/damage_vs_price(Q7).ipynb ` 
`Phase II Model&Analysis Path: ~/Phase II/Model_Q7_8_Shijie_Zhou/NeuralNetworkClassifier(Q7)p2.ipynb`

**Question 8:**  For used cars with different makes, will the accident record affect the used cars’ price differently? (Shijie Zhou)
`EDA path: ~/Phase I/EDA/damage_vs_price_make(Q8).ipynb ` 
`Phase II Model&Analysis Path: ~/Phase II/Model_Q7_8_Shijie_Zhou/Gaussian_Mixture_Model(Q8)p2.ipynb`





