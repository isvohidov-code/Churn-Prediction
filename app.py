from flask import Flask, render_template, request, send_file, flash, redirect, url_for, session
import pandas as pd
import pickle
import io
import os

# Load model, encoders, and scaler
with open('best_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('encoder.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler_data = pickle.load(scaler_file)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Required for flash messages

# Store the last bulk result in memory for download
_last_report = None


def make_prediction(input_data):
    """Single customer prediction."""
    input_df = pd.DataFrame([input_data])

    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    numerical_cols = ['tenure']
    input_df[numerical_cols] = scaler_data.transform(input_df[numerical_cols])

    prediction = loaded_model.predict(input_df)[0]
    probability = loaded_model.predict_proba(input_df)[0, 1]
    return "Churn" if prediction == 1 else "No Churn", probability


def make_bulk_prediction(df):
    """
    Predict churn for a full DataFrame.
    Returns the report DataFrame sorted by Churn_Probability descending.
    """
    df_report = df.copy()

    required_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'Contract', 'PaymentMethod']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if 'userId' not in df.columns:
        raise ValueError("Missing required column: userId")

    X = df[required_columns].copy()

    for col, encoder in encoders.items():
        if col in X.columns:
            X[col] = encoder.transform(X[col])

    numerical_cols = ['tenure']
    X[numerical_cols] = scaler_data.transform(X[numerical_cols])

    df_report['Churn_Prediction'] = loaded_model.predict(X)
    df_report['Churn_Probability'] = loaded_model.predict_proba(X)[:, 1]

    report_columns = ['userId', 'Churn_Prediction', 'Churn_Probability']
    final = df_report[report_columns].sort_values(by='Churn_Probability', ascending=False)
    return final


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    if request.method == 'POST':
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'Contract': request.form['Contract'],
            'PaymentMethod': request.form['PaymentMethod'],
        }
        prediction, probability = make_prediction(input_data)

    return render_template('index.html', prediction=prediction, probability=probability)


@app.route('/bulk', methods=['GET', 'POST'])
def bulk():
    global _last_report
    results = None

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            flash('No file selected. Please upload a CSV file.', 'error')
            return redirect(url_for('bulk'))

        if not file.filename.endswith('.csv'):
            flash('Invalid file type. Please upload a .csv file.', 'error')
            return redirect(url_for('bulk'))

        try:
            df = pd.read_csv(file)
            report = make_bulk_prediction(df)
            _last_report = report  # Cache for download
            results = report.to_dict(orient='records')
        except ValueError as e:
            flash(str(e), 'error')
            return redirect(url_for('bulk'))
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('bulk'))

    return render_template('bulk_predict.html', results=results)


@app.route('/bulk/download')
def bulk_download():
    global _last_report

    if _last_report is None:
        flash('No report available. Please run a prediction first.', 'error')
        return redirect(url_for('bulk'))

    output = io.StringIO()
    _last_report.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8-sig')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='client_risk_report.csv'
    )


if __name__ == '__main__':
    app.run(debug=True)
