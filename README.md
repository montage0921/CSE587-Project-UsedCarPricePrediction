# CSE587-Project-UsedCarPricePrediction Data Product

This readme.md provides an overview of the source code file structure for the Data Product for Phase III in the master branch. For details on the file structure for Phase I and Phase II, please refer to the readme.md in the main branch.

Team Number: 2  
Team Member:  
● 50608573 Te Shi teshi@buffalo.edu (git_id:montage0921)  
● 50388599 Shijie Zhou shijiezh@buffalo.edu  
● 50602483 Jiabao Yao jyao27@buffalo.edu  
● 50606804 Chao Wu cwu64@buffalo.edu

## File Descriptions

```
.
├── Database_Operation/   # Contains all database-related logic
│ ├── add_logic.py # Handles logic for adding records in database CRUD operations
│ ├── data_upload.py # Initializes connection to MySQL database and converts CSV files to SQL
│ ├── delete_logic.py # Handles logic for deleting records in database CRUD operations
│ ├── edit_logic.py # Handles logic for editing records in database CRUD operations
│ ├── search_result_logic.py # Handles logic for searching/looking up records in database CRUD operations
│ └── search_widgets_render.py # Renders search widgets for database CRUD operations
├── pages/ # Streamlit app pages
│ ├── 1-brand_analysis.py # Analysis of car brands
│ ├── 2-accident_history_prediction.py # Predicts accident history using models
│ ├── 3-regression_model_performance.py # Evaluates regression model performance
│ └── 4-admin.py # Admin page for CRUD operations
├── .gitignore # Specifies files and directories to be ignored by Git
├── README.md # Documentation for the master branch
├── config.yaml # Configuration file for database and admin credentials
├── homepage.py # Homepage script for the Streamlit application
├── price.png # Image used in the project
└── requirements.txt # List of dependencies required for the project
```
