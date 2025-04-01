from flask import Flask, render_template_string, request, jsonify
import numpy as np
import re
import random
import string
import joblib  # Import joblib for loading models

app = Flask(__name__)

# Load trained models and scaler
try:
    rf_model = joblib.load(r"C:\Users\shalu\Downloads\Password Analyzer\models\random_forest_model.pkl")
    xgb_model = joblib.load(r"C:\Users\shalu\Downloads\Password Analyzer\models\xgboost_model.pkl")
    scaler = joblib.load(r"C:\Users\shalu\Downloads\Password Analyzer\models\scaler.pkl")
    print("Models and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading models or scaler: {e}")
    rf_model = None
    xgb_model = None
    scaler = None

def calculate_entropy(password):
    """Calculate the Shannon entropy of the password."""
    if not password:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(password.count(chr(x))) / len(password)
        if p_x > 0:
            entropy += -p_x * np.log2(p_x)
    return entropy

def check_common_patterns(password):
    """Check for common password patterns."""
    patterns = {
        'sequential': r'(?:abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)',
        'repeated': r'(.)\1{2,}',
        'keyboard': r'(?:qwerty|asdfgh|zxcvbn)',
        'dates': r'\d{4}',
        'common_words': r'(?:password|admin|root|user|login|welcome)'
    }
    issues = []
    for pattern_name, pattern in patterns.items():
        if re.search(pattern, password.lower()):
            issues.append(f"Contains {pattern_name} pattern")
    return issues

def estimate_time_to_crack(password, attack_method="optimal", guesses_per_second=1e12):
    """Estimate time to crack based on attack method."""
    entropy = calculate_entropy(password)
    length = len(password)
    has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    has_numbers = bool(re.search(r'\d', password))
    has_uppercase = bool(re.search(r'[A-Z]', password))
    has_lowercase = bool(re.search(r'[a-z]', password))

    # Calculate the total number of possible combinations
    charset_size = 0
    if has_uppercase:
        charset_size += 26  # Uppercase letters
    if has_lowercase:
        charset_size += 26  # Lowercase letters
    if has_numbers:
        charset_size += 10  # Numbers
    if has_special:
        charset_size += len("!@#$%^&*(),.?\":{}|<>")  # Special characters

    # Total combinations = charset_size ^ length
    total_combinations = charset_size ** length

    # Adjust for attack method
    if attack_method == "dictionary":
        common_patterns_penalty = 0.1 if check_common_patterns(password) else 1.0
        leaked_dataset_penalty = 0.1 if password.lower() in {"123456", "password", "qwerty", "abc123"} else 1.0
        effective_combinations = total_combinations * common_patterns_penalty * leaked_dataset_penalty
        guesses_per_second *= 10  # Dictionary attacks are faster
    elif attack_method == "ai_based":
        ai_penalty = 0.5 if check_common_patterns(password) else 1.0
        effective_combinations = total_combinations * ai_penalty
        guesses_per_second *= 5  # AI-based attacks are moderately fast
    else:  # Default to optimal method
        optimal_penalty = 0.3 if check_common_patterns(password) else 1.0
        effective_combinations = total_combinations * optimal_penalty

    # Time to crack in seconds
    time_to_crack_seconds = effective_combinations / guesses_per_second

    # Simplify output for passwords with time-to-crack > 10 years
    if time_to_crack_seconds > 315360000:  # More than 10 years
        return "It will take years"

    # Convert to human-readable time
    if time_to_crack_seconds < 60:
        return f"Less than a minute"
    elif time_to_crack_seconds < 3600:
        return f"{int(time_to_crack_seconds / 60)} minutes"
    elif time_to_crack_seconds < 86400:
        return f"{int(time_to_crack_seconds / 3600)} hours"
    elif time_to_crack_seconds < 31536000:
        return f"{int(time_to_crack_seconds / 86400)} days"
    else:
        return f"{int(time_to_crack_seconds / 31536000)} years"

def validate_password(password):
    """Validate the password to ensure it meets all criteria."""
    issues = check_common_patterns(password)
    if len(password) < 12:
        issues.append("Password is too short")
    if not re.search(r'[A-Z]', password):
        issues.append("Missing uppercase letters")
    if not re.search(r'[a-z]', password):
        issues.append("Missing lowercase letters")
    if not re.search(r'\d', password):
        issues.append("Missing numbers")
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        issues.append("Missing special characters")
    return issues

def extract_features(password):
    """Extract features from a password for model prediction."""
    length = len(password)
    unique_chars = len(set(password))
    uppercase_count = sum(1 for char in password if char.isupper())
    lowercase_count = sum(1 for char in password if char.islower())
    digit_count = sum(1 for char in password if char.isdigit())
    special_count = sum(1 for char in password if not char.isalnum())
    return [length, unique_chars, uppercase_count, lowercase_count, digit_count, special_count]

def scale_model_prediction_to_100(prediction):
    """Scale the model's prediction to a range of 0-100."""
    MIN_MODEL_PREDICTION = 0.1  # Replace with actual minimum prediction
    MAX_MODEL_PREDICTION = 1.0  # Replace with actual maximum prediction
    if prediction < MIN_MODEL_PREDICTION:
        prediction = MIN_MODEL_PREDICTION
    elif prediction > MAX_MODEL_PREDICTION:
        prediction = MAX_MODEL_PREDICTION
    scaled_prediction = ((prediction - MIN_MODEL_PREDICTION) / (MAX_MODEL_PREDICTION - MIN_MODEL_PREDICTION)) * 100
    return round(scaled_prediction, 2)

def combine_model_and_entropy_strength(model_strength, entropy, password):
    """
    Combine the model's strength prediction with entropy, additional features, and dynamic penalties/bonuses.
    """
    # Scale entropy to a 0-100 range
    scaled_entropy = min(max((entropy / 10) * 100, 0), 100)  # Assuming max entropy is 10

    # Incorporate password length and unique character count as additional factors
    length_factor = min(len(password) * 2, 100)  # Scale length to a max of 100
    unique_chars = len(set(password))
    unique_factor = min(unique_chars * 5, 100)  # Scale unique characters to a max of 100

    # Add bonuses for complexity
    bonuses = 0
    if len(re.findall(r'[A-Z]', password)) > 1:  # Multiple uppercase letters
        bonuses += 5
    if len(re.findall(r'[a-z]', password)) > 1:  # Multiple lowercase letters
        bonuses += 5
    if len(re.findall(r'\d', password)) > 2:  # Multiple digits
        bonuses += 5
    if len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', password)) > 2:  # Multiple special characters
        bonuses += 5

    # Apply dynamic penalties for weak patterns and short passwords
    penalties = 0
    common_patterns = check_common_patterns(password)
    if common_patterns:
        penalties += len(common_patterns) * 5  # Deduct 5 points per common pattern
    if len(password) < 8:
        penalties += (8 - len(password)) * 3  # Deduct 3 points for each missing character below 8

    # Combine model strength, entropy, length, unique characters, bonuses, and penalties with weights
    combined_strength = (
        (model_strength * 0.4) +  # 40% weight for model prediction
        (scaled_entropy * 0.3) +  # 30% weight for entropy
        (length_factor * 0.2) +   # 20% weight for length
        (unique_factor * 0.1)     # 10% weight for unique characters
    ) + bonuses - penalties  # Add bonuses and subtract penalties

    # Ensure the final strength is within the range of 0-100
    return max(0, min(round(combined_strength, 2), 100))

def predict_password_strength(password):
    """
    Predict password strength using the trained models and combine it with entropy, additional factors, and penalties/bonuses.
    """
    if not rf_model or not xgb_model or not scaler:
        return "Model not available for prediction."

    # Extract features and scale them
    features = extract_features(password)
    scaled_features = scaler.transform([features])

    # Get predictions from both models
    rf_prediction = rf_model.predict(scaled_features)[0]
    xgb_prediction = xgb_model.predict(scaled_features)[0]
    average_prediction = (rf_prediction + xgb_prediction) / 2  # Average the predictions

    # Scale the model prediction to 0-100
    model_strength = scale_model_prediction_to_100(average_prediction)

    # Calculate entropy
    entropy = calculate_entropy(password)

    # Combine model strength, entropy, additional factors, and penalties/bonuses
    return combine_model_and_entropy_strength(model_strength, entropy, password)

def generate_random_password():
    """Generate a strong random password dynamically."""
    while True:
        length = random.randint(12, 16)  # Random length between 12 and 16
        characters = (
            random.choices(string.ascii_uppercase, k=3) +
            random.choices(string.ascii_lowercase, k=3) +
            random.choices(string.digits, k=3) +
            random.choices("!@#$%^&*(),.?\":{}|<>", k=3)
        )
        random.shuffle(characters)  # Shuffle to ensure randomness
        password = ''.join(characters)

        # Validate the generated password
        if not validate_password(password):  # Ensure it meets all criteria
            return password

def generate_suggestions(password):
    """Generate a solid password suggestion."""
    issues = validate_password(password)
    suggestions = []

    # If the password has no issues, return a positive message
    if not issues:
        suggestions.append("Keep it up! No more changes needed.")
        return suggestions

    # Provide actionable suggestions with examples
    if len(password) < 12:
        suggestions.append("Increase password length to at least 12 characters. Example: 'MySecurePass2023!'")
    if not re.search(r'[A-Z]', password):
        suggestions.append("Add uppercase letters. Example: 'securepass' -> 'SecurePass'")
    if not re.search(r'[a-z]', password):
        suggestions.append("Add lowercase letters. Example: 'SECUREPASS' -> 'SecurePass'")
    if not re.search(r'\d', password):
        suggestions.append("Add numbers. Example: 'SecurePass' -> 'SecurePass2023'")
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        suggestions.append("Add special characters. Example: 'SecurePass2023' -> 'SecurePass2023!'")

    # Generate random strongest passwords
    random_passwords = [generate_random_password() for _ in range(3)]
    suggestions.append("Random strongest passwords: " + ", ".join(random_passwords))

    return suggestions

@app.route('/')
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Password Strength Analyzer</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .container {
                max-width: 800px;
                margin-top: 2rem;
            }
            .card {
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border: none;
            }
            .card-header {
                background-color: #007bff;
                color: white;
                border-radius: 15px 15px 0 0 !important;
                padding: 1.5rem;
            }
            .password-input {
                border-radius: 10px;
                padding: 1rem;
                font-size: 1.2rem;
            }
            .strength-meter {
                height: 10px;
                border-radius: 5px;
                margin: 1rem 0;
                background-color: #e9ecef;
                overflow: hidden;
            }
            .strength-bar {
                height: 100%;
                width: 0;
                transition: width 0.3s ease-in-out;
            }
            .suggestion-item, .issue-item {
                padding: 0.5rem;
                margin: 0.5rem 0;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
            .issue-item {
                color: #dc3545;
                background-color: #fff5f5;
            }
            .metric-card {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 1rem;
                margin: 0.5rem 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <div class="card-header text-center">
                    <h2><i class="fas fa-shield-alt me-2"></i>AI Password Strength Analyzer</h2>
                    <p class="mb-0">Analyze your password strength with AI-powered insights</p>
                </div>
                <div class="card-body">
                    <div class="mb-4">
                        <label for="password" class="form-label">Enter your password:</label>
                        <div class="input-group">
                            <input type="password" class="form-control password-input" id="password" placeholder="Type your password here...">
                            <button class="btn btn-outline-secondary" type="button" id="togglePassword">
                                <i class="fas fa-eye"></i>
                            </button>
                        </div>
                    </div>

                    <div class="mb-4">
                        <label for="attackMethod" class="form-label">Select Attack Method:</label>
                        <select class="form-select" id="attackMethod">
                            <option value="optimal" selected>Optimal</option>
                            <option value="dictionary">Dictionary</option>
                            <option value="ai_based">AI-Based</option>
                        </select>
                    </div>

                    <div class="strength-meter">
                        <div class="strength-bar" id="strengthBar"></div>
                    </div>

                    <div class="text-center mb-4">
                        <h3 id="strengthText" class="mb-0">Predicted Strength: <span id="strengthScore">0</span>/100</h3>
                    </div>

                    <div class="row">
                        <div class="col-md-6">
                            <div class="metric-card">
                                <h5><i class="fas fa-clock me-2"></i>Time to Crack</h5>
                                <p id="timeToCrack" class="mb-0">-</p>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="metric-card">
                                <h5><i class="fas fa-random me-2"></i>Entropy</h5>
                                <p id="entropy" class="mb-0">-</p>
                            </div>
                        </div>
                    </div>

                    <div class="mt-4">
                        <h5><i class="fas fa-lightbulb me-2"></i>Suggestions:</h5>
                        <div id="suggestionsList"></div>
                    </div>

                    <div class="mt-4">
                        <h5><i class="fas fa-exclamation-circle me-2"></i>Issues Found:</h5>
                        <div id="issuesList"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let passwordInput = document.getElementById('password');
            let toggleButton = document.getElementById('togglePassword');
            let attackMethodSelect = document.getElementById('attackMethod');
            let strengthBar = document.getElementById('strengthBar');
            let strengthScore = document.getElementById('strengthScore');
            let timeToCrack = document.getElementById('timeToCrack');
            let entropy = document.getElementById('entropy');
            let suggestionsList = document.getElementById('suggestionsList');
            let issuesList = document.getElementById('issuesList');

            let timeout = null;

            toggleButton.addEventListener('click', function() {
                const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
                passwordInput.setAttribute('type', type);
                toggleButton.querySelector('i').classList.toggle('fa-eye');
                toggleButton.querySelector('i').classList.toggle('fa-eye-slash');
            });

            passwordInput.addEventListener('input', function() {
                clearTimeout(timeout);
                timeout = setTimeout(analyzePassword, 500);
            });

            attackMethodSelect.addEventListener('change', analyzePassword);

            function analyzePassword() {
                const password = passwordInput.value;
                const attackMethod = attackMethodSelect.value;

                if (!password) {
                    resetUI();
                    return;
                }

                fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ password: password, attack_method: attackMethod })
                })
                .then(response => response.json())
                .then(data => {
                    updateUI(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    resetUI();
                });
            }

            function updateUI(data) {
                // Update strength bar and score
                strengthBar.style.width = `${data.predicted_strength}%`;
                strengthBar.style.backgroundColor = getStrengthColor(data.predicted_strength);
                strengthScore.textContent = data.predicted_strength;

                // Update metrics
                timeToCrack.textContent = data.time_to_crack;
                entropy.textContent = data.entropy;

                // Update suggestions
                suggestionsList.innerHTML = '';
                data.suggestions.forEach(suggestion => {
                    const div = document.createElement('div');
                    div.className = 'suggestion-item';
                    div.innerHTML = `<i class="fas fa-check-circle me-2"></i>${suggestion}`;
                    suggestionsList.appendChild(div);
                });

                // Update issues
                issuesList.innerHTML = '';
                data.issues.forEach(issue => {
                    const div = document.createElement('div');
                    div.className = 'issue-item';
                    div.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>${issue}`;
                    issuesList.appendChild(div);
                });
            }

            function resetUI() {
                strengthBar.style.width = '0%';
                strengthScore.textContent = '0';
                timeToCrack.textContent = '-';
                entropy.textContent = '-';
                suggestionsList.innerHTML = '';
                issuesList.innerHTML = '';
            }

            function getStrengthColor(score) {
                if (score < 30) return '#dc3545';
                if (score < 60) return '#ffc107';
                if (score < 80) return '#17a2b8';
                return '#28a745';
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/analyze', methods=['POST'])
def analyze_password():
    data = request.get_json()
    password = data.get('password', '')
    attack_method = data.get('attack_method', 'optimal')  # Default to 'optimal'

    if not password:
        return jsonify({'error': 'No password provided'}), 400

    # Predict password strength using the models
    predicted_strength = predict_password_strength(password)

    # Calculate entropy and time to crack
    entropy = calculate_entropy(password)
    time_to_crack = estimate_time_to_crack(password, attack_method=attack_method)

    # Generate suggestions
    suggestions = generate_suggestions(password)

    # Validate password and get issues
    issues = validate_password(password)

    return jsonify({
        'predicted_strength': predicted_strength,
        'entropy': round(entropy, 2),
        'time_to_crack': time_to_crack,
        'suggestions': suggestions,
        'issues': issues
    })

if __name__ == '__main__':
    app.run(debug=True)
