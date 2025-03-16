# Machine Learning Regression with Neural Network, XGBoost, and Random Forest  

## Dataset  
- Spectral reflectance data with 448 features and 500 Rows.  
- Target variable: Vomitoxin levels (in ppb).  

## Overview  
This project builds and compares multiple regression models to predict vomitoxin levels using high-dimensional spectral reflectance data. Models include a Neural Network (TensorFlow + Keras), XGBoost, and Random Forest. SHAP (SHapley Additive exPlanations) was used to interpret the predictions and identify the most influential features. The final model was deployed to **Azure** using **Docker**.  

## Workflow  

### 1. **Preprocessing**  
- Cleaned data and handled missing values.  
- Applied Box-Cox and PowerTransformer to reduce skewness and improve normality.  
- Scaled features using `StandardScaler` for consistency.  
---

### 2. **Model Building**  
- **Neural Network:**  
   - Multi-layered architecture with dense layers and dropout.  
   - Adam optimizer used for training.  

- **XGBoost:**  
   - Tree-based boosting algorithm.  
   - Tuned for learning rate, depth, and estimators.  

- **Random Forest:**  
   - Ensemble of decision trees.  
   - Tuned for number of trees and depth.  

---

### 3. **Hyperparameter Tuning**  
- Used **Optuna** for Neural Network tuning.  
- Used **RandomizedSearchCV** and **GridSearchCV** for XGBoost and Random Forest.  
- Early stopping applied to prevent overfitting.  

---

### 4. **Model Evaluation**  
- Evaluated using:  
   - **Mean Absolute Error (MAE)**  
   - **Root Mean Squared Error (RMSE)**  
   - **R²**  
- Neural Network showed the highest accuracy.  

---

### 5. **SHAP Analysis**  
- SHAP identified the most influential features.  
- High feature values were linked to increased predictions.  

---

### 6. **Model Deployment**  
- Final model was containerized using **Docker**.  
- Deployed to **Azure** using **FastAPI** for serving predictions.  
- API endpoints were created for easy model access and scalability.

### Application link
- Azure link: https://imagoai-app.azurewebsites.net/docs

