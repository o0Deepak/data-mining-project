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

