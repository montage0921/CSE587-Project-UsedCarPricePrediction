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

## Build the App from source code

1. **Clone or Download the Repository**  
   Clone the master branch using Git or download the project as a ZIP file:

   ```bash
   git clone <repository-url>
   ```

2. Install Required Libraries
   Navigate to the project directory and install the necessary dependencies using the following command:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Application
   Open a terminal, navigate to the project directory, and use the following command to start the application:
   ```bash
   streamlit run homepage.py
   ```
   The application is also deployed online, powered by Render. You can access the live version of the app at the following link:

https://cse587-project-usedcarpriceprediction.onrender.com
