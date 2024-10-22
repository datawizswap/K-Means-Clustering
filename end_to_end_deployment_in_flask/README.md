# KMeans Clustering for University Data

### Overview
This project implements a KMeans clustering algorithm on university data to group universities based on features like SAT scores, acceptance rates, and more. The project includes data preprocessing (handling missing values, scaling), clustering analysis (using KMeans), and a Flask-based web application for dynamic university clustering. The results provide insight into how universities can be grouped based on their characteristics.

Features
Data Preprocessing: Handling missing values, scaling numerical data, and encoding categorical variables.
KMeans Clustering: Implementation of the elbow method and silhouette score to determine optimal clusters.
Model Persistence: Saving pipelines and models using Joblib and Pickle.
Flask Web Application: A simple web interface that allows users to upload university data and get clustering predictions.
PostgreSQL Integration: Reading and writing data to a PostgreSQL database using SQLAlchemy.
Technologies Used
Python: Core programming language for the project.
Pandas: Data manipulation and analysis.
Scikit-learn: Machine learning library for clustering and evaluation.
Sweetviz: For automated exploratory data analysis.
PostgreSQL & SQLAlchemy: For database interaction.
Flask: Web framework for the application.
Joblib & Pickle: For saving and loading models.
Matplotlib: For data visualization (elbow plot).
KneeLocator: For detecting the optimal number of clusters.
Project Structure
plaintext
Copy code
University_KMeans_Clustering/
├── data/
│   └── University_Clustering.xlsx  # Dataset used for clustering
├── src/
│   ├── app.py                      # Flask application
│   ├── clustering_pipeline.py       # Data preprocessing and clustering code
│   ├── model_saving.py              # Code to save and load models
│   └── templates/
│       └── index.html               # HTML template for Flask app
├── static/
│   └── plots/                       # Folder containing generated plots (e.g., elbow curve)
├── saved_models/
│   └── clust_univ.pkl               # KMeans model
│   └── processed1                   # Imputation and one-hot encoding pipeline
│   └── processed2                   # Scaling pipeline
├── README.md                        # Project overview
└── requirements.txt                 # Python dependencies
Data Preprocessing
Loading the Dataset: Read data from PostgreSQL using SQLAlchemy.
Handling Missing Data: Impute missing values using mean imputation.
Scaling Numerical Data: Apply Min-Max scaling to numerical columns for consistency in clustering.
Saving Preprocessing Pipelines: Save the imputation and scaling pipelines for future use.
python
Copy code
# Imputation and scaling
num_pipeline = Pipeline([('impute', SimpleImputer(strategy='mean')), ('scale', MinMaxScaler())])
processed = num_pipeline.fit(df1[numeric_features])

# Save pipeline
joblib.dump(processed, 'processed1')
KMeans Clustering
Elbow Method: The optimal number of clusters is determined using the elbow plot.
KMeans Algorithm: Implement the KMeans clustering algorithm with the optimal number of clusters.
Cluster Evaluation: Evaluate clusters using silhouette score, Calinski-Harabasz, and Davies-Bouldin index.
python
Copy code
# Elbow method to determine the number of clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(univ_clean)
Flask Web Application
Web Interface: A simple web app built using Flask, allowing users to upload Excel files and get clustering predictions for new university data.
Model Integration: Load pre-trained models and pipelines to preprocess data and predict clusters dynamically.
python
Copy code
@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        data = pd.read_excel(f)
        # Preprocessing
        data_clean = preprocess_data(data)
        prediction = model1.predict(data_clean)
        return render_template('index.html', Z="Clusters for new university data:", Y=prediction)
Installation & Usage
Requirements
Python 3.x
PostgreSQL
Install dependencies using:
bash
Copy code
pip install -r requirements.txt
Running the Project
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/university_kmeans_clustering.git
Navigate to the project folder:
bash
Copy code
cd University_KMeans_Clustering
Start the Flask application:
bash
Copy code
python src/app.py
Access the web interface by visiting http://localhost:5000/ in your browser.
Using the Application
Upload an Excel file with university data (excluding UnivID and Univ columns).
The application will return the predicted clusters for the uploaded data.
Results
The elbow method and silhouette scores show that 2 clusters provide the best fit for the dataset.
Clusters are evaluated, and labels are assigned to each university based on the clustering results.
The clustering results are saved in a CSV file (University.csv), and a report is generated using Sweetviz for further exploration.
Conclusion
This project successfully implements KMeans clustering on university data with a web-based interface for interactive clustering analysis. It demonstrates how to preprocess data, build clustering models, and deploy them in a web application.

