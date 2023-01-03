# API Dependencies
import sys
sys.path.append('../') #to add the directory containing the config script
import maon_config as config

import numpy as np
import pandas as pd
import psycopg2
from flask import Flask
from sklearn.ensemble._forest import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def connect():
    
    # Set up a connection to the postgres server.
    print("Setting up the postgres server...")
    conn_string = "host="+ config.PGHOST  +" dbname="+ config.PGDATABASE +" user=" + config.PGUSER \
                  +" password="+ config.PGPASSWORD
    #print(conn_string)
    print("\t...completed\nConnecting to the server...")
    conn = psycopg2.connect(conn_string)
    print("\t...Server Connected!")
    print("Importing data...")
    cursor = conn.cursor()
    
    # Create a cursor object
    cursor.execute("SELECT * FROM maontech_dataset")
    data = cursor.fetchall()
    print("\t...data imported!")
    return data

data = connect()
columns = ["region", "depot", "item_no", "tms", "ams", "month", "year"]
df = pd.DataFrame(data, columns=columns)
del columns


app = Flask(__name__)

def data_preprocessing(df):
    """Private helper function to preprocess data for model prediction.
    All the codes required for feature engineering/selection are defined here.
    Parameters
    ----------
    df : str
        The data payload received within POST requests sent to our API.
    Returns
    -------
    X_data : A dataframe of all relevant features asides the target variable

    y_data : the target variable; 'ams'.
    """
    df = df[df['region']=="SW"]
    df = df.drop(['region', 'tms', 'year'], axis = 1)#, inplace=True)
    df = df.sort_values(by='month')
    df['ams'] = abs(df['ams'])
    df_dummies = pd.get_dummies(df)#, drop_first=True)

    #reindex the columns to make the target variable the last
    cols = [col for col in df_dummies.columns if col != 'ams'] + ['ams']

    df_dummies = df_dummies.reindex(columns=cols)
    
    X_data = df_dummies[[col for col in df_dummies.columns if col != "ams"]]
    y_data = df_dummies['ams']

    return X_data, y_data

def _model(df):
    X, y = data_preprocessing(df)
    print("Done processing data...")
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=42,
                                                        shuffle=False
                                                        )
    rfr = RandomForestRegressor(random_state=42)
    print("Fitting model...")
    rfr.fit(X_train, y_train)
    
    print("Running prediction")
    result = []
    y_pred = rfr.predict(X_test) #.score(X_train, y_train)
    MSE = mean_squared_error(y_test, y_pred)
    rmse = round(np.sqrt(MSE), 2)
    result.append('RandomForestRegression RMSE: SW')
    result.append(rmse)
    accuracy_ = round(r2_score(y_test, y_pred) , 2)
    #print(f"Accuracy: {accuracy_}%")
    result.append(f"Accuracy: {accuracy_}")
    
    return result

predictions = _model(df)

@app.route('/')
def prediction():
    print(predictions)
    return predictions

if __name__ == '__main__':
   app.run(debug=True)