

https://github.com/user-attachments/assets/9a1a478f-8932-45af-a91d-91a5417cf0fa

Here's the updated version with your requested edits:  

---

# AI Password Strength Analyzer  

An intelligent tool that analyzes password strength using AI and machine learning techniques to predict password vulnerability and provide actionable suggestions for improvement.  

## Features  

- Real-time password strength analysis  
- AI-powered password suggestions  
- Time-to-crack estimation  
- Entropy calculation  
- Pattern detection for common password vulnerabilities  
- Modern, responsive UI  
- Secure password handling (no storage)  

## Requirements  

- Python 3.8+  
- Flask  
- TensorFlow  
- Transformers (Hugging Face)  
- Other dependencies listed in `requirements.txt`  

## Installation  

1. Clone the repository:  
   ```bash
   git clone <repository-url>
   cd password-strength-analyzer
   ```
2. Create a virtual environment (recommended):  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## Usage  

1. Start the Flask application:  
   ```bash
   python ML_Model_app.py
   ```
2. Open your web browser and navigate to:  
   ```
   http://localhost:5000
   ```
3. Enter a password in the input field to analyze its strength.  

## How It Works  

The tool uses multiple techniques to analyze password strength:  

1. **Entropy Calculation**: Measures the randomness and unpredictability of the password  
2. **Pattern Detection**: Identifies common patterns that make passwords vulnerable  
3. **Time-to-Crack Estimation**: Calculates approximate time needed to crack the password  
4. **AI-Powered Analysis**: Uses machine learning models trained on password datasets  
5. **Smart Suggestions**: Provides specific recommendations for improvement  

## Files Overview  

- **ML_Model_app.py** – Uses a machine learning approach with **Random Forest Classifier** and **XGBoost** to predict password strength.  
- **standalone_app.py** – Implements a rule-based approach to assess password strength without using ML models.  
- **model_training.py** – Trains the **Random Forest Classifier** and **XGBoost** models using feature extraction techniques. These trained models are then used for password strength prediction.  
- **Saved Model Files**:  
  - `scaler.pkl` – Stores the feature scaler used for model training.  
  - `random_forest_classifier.pkl` – Pre-trained Random Forest model for password strength prediction.  
  - `xgb_boost.pkl` – Pre-trained XGBoost model for password strength prediction.  

## Security Note  

This tool is designed for password strength analysis only. It does not store or transmit passwords to any external servers. All analysis is performed locally in your browser and on your machine.  

## Contributing  

Contributions are welcome! Please feel free to submit a Pull Request.  




