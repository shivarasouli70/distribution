import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf 
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
import dash
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
#Import dataset
#df=pd.read_csv("Desktop/Semester2/IMSE586/Project586-Team17/data/Boston.csv")
df=pd.read_csv("./data/Boston.csv")
#Drop column 1, which is labeled as 'Unnamed,' and column 'black,' as it contains offensive content related to racism
df= df.drop(df.columns[[0,12]], axis=1)
#Drop column 'nox' and 'rad' randomly, based on the high correlation with other variables in heatmap
df= df.drop(df.columns[[4,8]], axis=1)
# Add a new column to DataFrame, which is a categorical variable about house price
df = df.assign(price=lambda x: np.where(x.iloc[:, 10] >= 25, 'high', 'low'))
# Make price dummy
df = df.assign(price_binary=lambda df: pd.get_dummies(df['price'], drop_first=True, dtype=int))
# Initialize dashboard
app = Dash(__name__)
server=app.server
app.layout = html.Div([
    html.H1("Boston Housing Data Analysis"),   # dashboard title
    html.H2("Select a Feature:"),
    dcc.Dropdown(
        id='feature',
        options=list(df.columns),   #in this dropdown menue, we have the name of columns (features) as  our options
        value=df.columns[0]   # consider the first column (crim) as defined value before selecting other options
    ),
    html.Div(id='statistical_output'),   #get statistical summary of each feature 
    dcc.Graph(id='histogram'),   # get the distribution plot 
])

@app.callback(
    [Output('statistical_output', 'children'),
     Output('histogram', 'figure')],
    [Input('feature', 'value')]
)
def update_output(selected_feature):
   
    selected_data = df[selected_feature]
    Minimum= round(selected_data.min(),2)  #statistical output Min, Max, Mean, Std
    Maximum=round(selected_data.max(),2)
    Mean=round(selected_data.mean(),2)
    Standard_Deviation= round(selected_data.std(),2)
    
    # Print statistical_output
    statistical_output = html.Div([
            html.H2("Statistical Summary of Selected Feature:"),
            html.P(f"Minimum: {Minimum}"),
            html.P(f"Maximum=: {Maximum}"),
            html.P(f"Mean: {Mean}"),
            html.P(f"Standard Deviation: {Standard_Deviation}"), 

        ])

    # Create histogram
    histogram = px.histogram(df, x=selected_feature, title="Selected Feature Distribution")

    return statistical_output, histogram
#run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8040)
