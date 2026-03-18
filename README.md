# Customer Churn Prediction App

💼 **Description**  
This project is a web app that predicts whether a customer is likely to churn (leave) using an Artificial Neural Network (ANN) model. Users can input customer details, and the app provides an instant prediction.  

---

## 🛠 Features

- User-friendly web interface using **Streamlit**  
- Inputs include:
  - Credit Score
  - Age
  - Tenure
  - Balance
  - Number of Products
  - Credit Card ownership
  - Active membership
  - Estimated Salary
  - Geography (Germany, Spain)
  - Gender (Male)  
- Predicts customer churn based on ANN model  
- Visual feedback for churn/stay prediction  

---

## ⚡ Tech Stack

- Python 3.14  
- Streamlit  
- TensorFlow / Keras  
- Scikit-learn (for preprocessing & scaling)  
- NumPy  

---

## 🚀 How to Run

### Locally:

1. Clone the repo:  
   ```bash
   git clone https://github.com/faizih111/customer-churn-ANN.git
   cd customer-churn-ANN
2. Create a virtual environment:

python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
3. Install dependencies:

pip install -r requirements.txt
4. Run the app:

streamlit run app.py

On Streamlit Cloud:

Visit the deployed app:
[Customer Churn Prediction App](https://customer-churn-ann-aykqrj6scfdztq7stfky9h.streamlit.app/)

🧠 How it Works

1.Users input customer details on the sidebar.

2.Inputs are scaled using a saved scaler.

3.Scaled features are fed into the trained ANN model (model.h5).

4.The app predicts churn probability and displays the result.
