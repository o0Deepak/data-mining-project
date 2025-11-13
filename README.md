PROJECT OVERVIEW
-------------------
This project performs **Customer Segmentation** using multiple clustering algorithms:
- K-Means Clustering
- DBSCAN (Density-Based Spatial Clustering)
- Agglomerative (Hierarchical) Clustering

It also applies **Principal Component Analysis (PCA)** for dimensionality reduction
and includes **Age** and **Gender** as part of the feature set for analysis.

The purpose of this project is to identify distinct customer groups based on 
their Age, Gender, Annual Income, and Spending Score.

REQUIREMENTS
---------------
Ensure that Python 3.x is installed on your system.
The following Python libraries are required:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

You can install all required packages using this command:

pip install pandas numpy matplotlib seaborn scikit-learn scipy

HOW TO RUN THE PROJECT
-------------------------
1. Place both files in the same folder:
   - Mall_Customers.csv
   - customer_segmentation_fixed_v2.py

2. Open a terminal or command prompt in that folder.

3. Run the script using:
   python customer_segmentation_fixed_v2.py

4. The script will:
   - Load and preprocess the dataset
   - Apply PCA
   - Perform KMeans, DBSCAN, and Agglomerative clustering
   - Display multiple plots for each algorithm
   - Print summaries and cluster interpretations in the terminal

PRECAUTIONS & TROUBLESHOOTING
---------------------------------
- Ensure the dataset file is named exactly: Mall_Customers.csv
- Do NOT open the CSV file while running the script (it may cause read errors)
- If you get `KeyError` or `InvalidIndexError`, recheck that the dataset columns are:
  ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
- If plots do not appear all at once, ensure non-blocking mode (`plt.show(block=False)`) is active.
- For best performance, run in a Python IDE (VS Code, PyCharm, Jupyter, etc.)
- If DBSCAN produces few or no clusters, adjust parameters:
  eps=5, min_samples=5 in the code.


EXPECTED RESULTS
-------------------
You should see visually distinct customer clusters.
The terminal will also print segment summaries such as:

Cluster 0: Low income, low spending (Budget Customers)
Cluster 1: High income, high spending (Premium Customers)
Cluster 2: Average income, average spending (Mid-Tier)
Cluster 3: Low income, high spending (Potential Target)
Cluster 4: High income, low spending (Careful Spenders)


METHOD TO FOLLOW FOR CREATING A NEW DATASET
---------------------------------------------

1. FILE NAME AND TYPE
------------------------------------------------------------
- The dataset must be named exactly:
  Mall_Customers.csv

- Save it in the same folder as the Python project file.

- File type must be:
  CSV (Comma-Separated Values)

------------------------------------------------------------
2. COLUMN NAMES (IMPORTANT)
------------------------------------------------------------
The CSV file must contain these exact column headers:

CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)

Example row:
1,Male,25,45,77

------------------------------------------------------------
3. COLUMN DETAILS AND RULES
------------------------------------------------------------
• CustomerID:
  - Integer values (1, 2, 3, …)
  - Must be unique for each record.

• Gender:
  - Only “Male” or “Female”
  - Case-insensitive, but keep consistent.

• Age:
  - Integer values (18–70)
  - Must not contain blanks or text.

• Annual Income (k$):
  - Integer or float value
  - Represents income in thousands (e.g., 40 = $40,000)

• Spending Score (1-100):
  - Integer between 1 and 100
  - Indicates spending behavior (higher = more spending)

------------------------------------------------------------
4. SAMPLE DATA FORMAT
------------------------------------------------------------
CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,25,45,77
2,Female,31,32,40
3,Male,22,50,95
4,Female,45,80,10
5,Male,29,60,65
6,Female,34,35,47

------------------------------------------------------------
5. COMMON MISTAKES TO AVOID
------------------------------------------------------------
 Do not rename or remove any column headers.
 Do not include extra spaces or symbols in headers.
 Do not leave empty cells.
 Do not save as .xls or .xlsx — only .csv.
 Do not include special characters like ₹, $, or commas in numbers.

------------------------------------------------------------
6. CHECKING YOUR DATA
------------------------------------------------------------
To verify your dataset:
- Open in a text editor and confirm it looks like this:

CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,25,45,77
2,Female,31,32,40
3,Male,22,50,95
...

------------------------------------------------------------
7. OPTIONAL – AUTO-GENERATE A VALID DATASET
------------------------------------------------------------
You can use this short Python script to generate a valid dataset automatically:

import pandas as pd
import numpy as np

np.random.seed(42)
n = 200  # number of customers

data = pd.DataFrame({
    'CustomerID': range(1, n+1),
    'Gender': np.random.choice(['Male', 'Female'], size=n),
    'Age': np.random.randint(18, 70, size=n),
    'Annual Income (k$)': np.random.randint(15, 150, size=n),
    'Spending Score (1-100)': np.random.randint(1, 100, size=n)
})

data.to_csv('Mall_Customers.csv', index=False)
print("Mall_Customers.csv created successfully.")

------------------------------------------------------------
8. FINAL CHECK
------------------------------------------------------------
 File name: Mall_Customers.csv
 Columns: 5 (exactly as listed)
 Values: Numeric (except Gender)
 Format: CSV
 Location: Same directory as your Python file

Once this structure is followed, your code will run smoothly
and generate all plots and outputs correctly.

