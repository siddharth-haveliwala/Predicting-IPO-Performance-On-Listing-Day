# 🚀 Predicting IPO Performance on Listing Day

This project delivers a robust Machine Learning solution to predict the performance of Initial Public Offerings (IPOs) on their listing day in the Indian Stock Market. By analyzing critical pre-listing financial and subscription metrics, the model classifies an IPO as a _'Success'_ or _'Failure'_ (i.e., whether it lists at a premium or a discount).

The predictive model is containerized and deployed as a live web application using **Flask**, allowing users to input real-time IPO metrics and receive an instant prediction.

**Refer to the Project Presentation.pptx for project overview.**

## ✨ Key Features
- **Classification Model:** Utilizes a **Logistic Regression Classifier** (chosen for its high accuracy and explainability) to predict the listing-day outcome.

- **High Accuracy:** The finalized model achieved an accuracy of 76.67% on the test dataset.

- **Web Deployment:** A user-friendly web interface built with Flask and HTML/CSS allows for real-time inference.

- **API Ready:** Includes a clean backend structure for integration into other applications.

## 🎯 Model & Methodology

### Data Features Used
The model uses the following seven key features, which capture both the financial valuation and market demand of the IPO:


| Feature Name | Description |
|--------------|-------------|
| **Issue Price** | The final price per share offered to investors. |
| **Lot Size** | The minimum number of shares an investor must bid for. |
| **Issue Price (Rs Cr)** | The total issue size in terms of valuation (in Crores). |
| **QIB (Qualified Institutional Buyer)** | Subscription rate by Qualified Institutional Buyers. |
| **NII (Net Individual Investors)** | Subscription rate by Non-Institutional Investors. |
| **TOTAL** | The total overall subscription rate. |

### Model Performance
Multiple classification models were evaluated during the exploration phase (`model.py`). The best-performing models, achieving the highest accuracy, were:

| Model | Accuracy (%) |
|--------------|-------------|
| **Logistic Regression** | 76.7% |
| **Random Forest Classifier** | 76.7% |
| **Multilayer Perceptron (NN)** | 76.7% |

The Logistic Regression model was selected for final deployment due to its superior interpretability and efficiency, making the prediction results more transparent.

## ⚙️ Project Architecture
The project follows a standard MLOps-lite architecture:

- Training (`model.py` / `ipo_prediction.ipynb`): Data preprocessing, feature selection, model training, and hyperparameter tuning. The final model is serialized and saved as `model.pkl`.

- Serving (`app.py`): A lightweight Flask server loads the `model.pkl` file into memory upon startup.

- Prediction API Endpoint (`/predict`): The server receives input data from the web form (`index.html`), transforms it into the format expected by the model, makes a prediction, and returns the result to the user.

## ▶️ Getting Started (Local Setup)
To run the application locally on your machine, follow these steps:

**1. Clone the Repository**
```
git clone [https://github.com/YourUsername/ipo-performance-predictor.git](https://github.com/YourUsername/ipo-performance-predictor.git)
cd ipo-performance-predictor
```

**3. Set up Virtual Environment**
It is highly recommended to use a virtual environment to manage dependencies.

```
# Create a virtual environment
python -m venv venv
# Activate the environment (Linux/macOS)
source venv/bin/activate
# Activate the environment (Windows)
.\venv\Scripts\activate
```

**3. Install Dependencies**
Install all necessary Python packages using the provided `requirements.txt`.

```
pip install -r requirements.txt
```

**4. Run the Flask Application**
Start the web server:

```python app.py```

The application will now be running on `http://127.0.0.1:5000/`. Open this URL in your browser to access the prediction interface.

## ☁️ Deployment (Heroku)
This project is configured for deployment on the Heroku cloud platform.

- **Prerequisites**: Ensure you have the Heroku CLI installed and logged in.

- **Create Heroku App**:

`heroku create your_app_name`

- **Push to Heroku**:

```
git add .
git commit -m "Final version for Heroku deployment"
git push heroku main
```

The Profile and `requirements.txt` will automatically instruct Heroku to set up a Gunicorn server and deploy your application. The app will be available at `your_app_name.herokuapp.com`.
