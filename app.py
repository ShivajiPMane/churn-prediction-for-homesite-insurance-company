from flask import Flask, render_template, request
import pandas as pd
import joblib
from pre_processing import Preprocess


app = Flask(__name__)

model = joblib.load("files/xgb_clf_model.pkl")


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("home.html")


@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        f = request.files['file']
        file = pd.read_csv(f)
        pp = Preprocess()
        data_pt = pp.preprocessing_datapoint(file)
        prediction = model.predict(data_pt)[0]

    if prediction == 1:
        op_str = "Churned : This Customer is likely tobe purchased a quoted insurance plan."
    else:
        op_str = "Not Churned : This Customer is might not purchase a quoted insurance plan"

    return render_template("data.html", data=op_str)


if __name__ == "__main__":
    app.run(debug=True)
