from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model/titanic_survival_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            pclass = int(request.form['Pclass'])
            sex = int(request.form['Sex'])  # 0=Female, 1=Male
            age = float(request.form['Age'])
            sibsp = int(request.form['SibSp'])
            fare = float(request.form['Fare'])
            features = np.array([[pclass, sex, age, sibsp, fare]])
            pred = model.predict(features)[0]
            result = 'Survived' if pred == 1 else 'Did Not Survive'
        except Exception as e:
            result = f'Error: {e}'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
