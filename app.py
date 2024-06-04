from flask import Flask,request, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import plotly.express as px
plt.rcParams['figure.figsize']=(16,5)
plt.style.use('fivethirtyeight')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def helloworld():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():

#////////////////////////////////////////////

    # Load the dataset
    data = pd.read_csv('insurance.csv')
    print(data.head())

    data['sex'] = data['sex'].map({'male': 1, 'female': 2})
    data['smoker'] = data['smoker'].map({'yes': 1, 'no': 2})
    data = data.drop('region', axis=1)

    # Check for missing values
    print(data.isnull().sum())

    # Summary statistics
    print(data.describe())

    # Distribution of numerical features
    numerical_features = ['age', 'bmi', 'children', 'charges']
    data[numerical_features].hist(bins=15, figsize=(15, 6), layout=(2, 2))
    # plt.show()

    # Distribution of categorical features
    categorical_features = ['sex', 'smoker']
    for feature in categorical_features:
        sns.countplot(x=feature, data=data)
        # plt.show()

    # Pairplot to see relationships
    sns.pairplot(data)
    # plt.show()

    # Defining input features (X) and output target (y)
    X = data.drop('charges', axis=1)  # Input features
    y = data['charges']  # Output target

    # Defining the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['age', 'bmi', 'children']),
            ('cat', OneHotEncoder(), ['sex', 'smoker'])
        ])

    # Create preprocessing and modeling pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # Correlation matrix
    corr_matrix = data.corr()
    print(corr_matrix)

    # Visualizing the correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    # plt.show()

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the model
    pipeline.fit(X_train, y_train)

    # Making predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Evaluating the model
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f'Train MSE: {mse_train:.2f}, Test MSE: {mse_test:.2f}')
    print(f'Train R^2: {r2_train:.2f}, Test R^2: {r2_test:.2f}')

    # Function to predict charges based on user input
    def predict_charges(age, sex, bmi, children, smoker):
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker]
        })

        prediction = pipeline.predict(input_data)
        return prediction[0]


    if request.method == 'POST':
        # Retrieve the text from the textarea
        age2 = request.form.get('age')
        sex2 = request.form.get('gender')
        bmi2 = request.form.get('bmi')
        children2 = request.form.get('children')
        smoker2 = request.form.get('smoker')

    # Example usage
    age = age2
    sex = (int)(sex2)
    bmi = (float)(bmi2)
    children = (int)(children2)
    smoker = (int)(smoker2)


    predicted_charges = predict_charges(age, sex, bmi, children, smoker)

    if (predicted_charges < 0):
        predicted_charges = 0

    a = f'Your Predicted Charges: ${predicted_charges:.2f}'




#//////////////////////////////////////////////////////////////


    return a

if __name__ == '__main__':
    app.run(port=3000, debug=True)