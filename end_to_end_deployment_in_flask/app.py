from flask import Flask, render_template, request
from sqlalchemy import create_engine
import pandas as pd
import pickle
import joblib

# Load the preprocessed pipeline and KMeans model
processed1 = joblib.load('processed1')  # Min-max scaling and imputation pipeline
model = pickle.load(open('Clust_Univ.pkl', 'rb'))  # KMeans clustering model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        user = request.form['user']
        pw = request.form['password']   # Extracting password from the form
        host = request.form['host']     # Extracting host from the form
        port = request.form['port']     # Extracting port from the form
        db = request.form['db']

        # Create the connection string
        engine = create_engine(f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}")

        try:
            # Attempt to read the file (either CSV or Excel)
            data = pd.read_csv(f)
        except:
            try:
                data = pd.read_excel(f)
            except Exception as e:
                return str(e)
                
        # Drop unwanted features
        univ_df = data.drop(["UnivID", "Univ"], axis=1)
        
        # Process numeric features
        numeric_features = univ_df.select_dtypes(exclude=['object']).columns
        data1 = pd.DataFrame(processed1.transform(univ_df[numeric_features]), columns=numeric_features)
        
        # Predict clusters
        prediction = pd.DataFrame(model.predict(data1), columns=['cluster_id'])
        prediction = pd.concat([prediction, data], axis=1)
        
        # Save prediction to the SQL database
        prediction.to_sql('university_pred_kmeans', con=engine, if_exists='replace', chunksize=1000)
        
        # Create HTML table
        html_table = prediction.to_html(classes='table table-striped')
        
        return render_template("data.html", Y=f"<style>\
                               .table {{\
                                    width: 50%;\
                                    margin: 0 auto;\
                                    border-collapse: collapse;\
                                    }}\
                               .table thead {{\
                                    background-color:#39648f;\
                                    }}\
                               .table th, .table td{{\
                                    border: 1px solid #ddd;\
                                    padding:8px;\
                                    text-align:center;\
                                    }}\
                               .table td {{\
                                    background-color: #888a9e;\
                                    }}\
                               .table tbody th {{\
                                    background-color:#ab2c3f;\
                                    }}\
                               </style>\
                               {html_table}")

if __name__ == '__main__':
    app.run(debug=True)
