from datetime import timedelta

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from BestTimeToFertilizeModule import BestTimeToFertilize
from NPKEstimatorModule import NPKEstimator

from models import db, User, Notification, Product
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from mandi_price_module import get_mandi_prices
# ================= ML IMPORTS ================= #
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import re
from werkzeug.utils import secure_filename
# ================= PDF ================= #
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ================= AI CHATBOT (DEEPSEEK VIA NVIDIA ENDPOINT) ================= #
from openai import OpenAI


ai_client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = "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"
)

app = Flask(__name__)
app.secret_key = "secret123"

# ================= DATABASE ================= #
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# ================= GLOBAL REPORT STORAGE ================= #
latest_report = {}

# ================= LOAD MODEL ================= #
model = tf.keras.models.load_model("disease_model.h5")

# ================= SOIL HEALTH MODELS ================= #
import joblib

soil_model = joblib.load("soil_type_model.pkl")
soil_le = joblib.load("soil_type_label_encoder.pkl")

variety_model = joblib.load("variety_model.pkl")
variety_le = joblib.load("variety_label_encoder.pkl")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_labels = {v: k for k, v in class_indices.items()}

# ================= DISEASE INFO ================= #

disease_info = {

    # 🍎 APPLE
    "Apple___Apple_scab": {
        "description": "Fungal disease causing dark, scabby lesions on leaves and fruits.",
        "solution": "Apply fungicides like captan or sulfur and remove infected leaves."
    },
    "Apple___Black_rot": {
        "description": "Causes black spots on leaves and fruit rot.",
        "solution": "Prune infected parts and use fungicides."
    },
    "Apple___Cedar_apple_rust": {
        "description": "Rust disease forming orange spots on leaves.",
        "solution": "Remove nearby cedar trees and apply fungicide."
    },
    "Apple___healthy": {
        "description": "Plant is healthy with no visible disease.",
        "solution": "Maintain proper care and monitoring."
    },

    # 🌶️ BELL PEPPER
    "Bell_Pepper___Bacterial_spot": {
        "description": "Bacterial disease causing dark spots on leaves and fruit.",
        "solution": "Use copper-based sprays and disease-free seeds."
    },
    "Bell_Pepper___healthy": {
        "description": "Healthy plant with no infection.",
        "solution": "Maintain proper irrigation and care."
    },

    # 🍒 CHERRY
    "Cherry___Powdery_mildew": {
        "description": "White powdery fungal growth on leaves.",
        "solution": "Apply sulfur fungicide and improve air circulation."
    },
    "Cherry___healthy": {
        "description": "Healthy plant.",
        "solution": "No treatment required."
    },

    # 🌽 CORN
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Gray lesions on leaves caused by fungus.",
        "solution": "Use resistant varieties and fungicides."
    },
    "Corn___Common_rust": {
        "description": "Reddish-brown pustules on leaves.",
        "solution": "Apply fungicide and use resistant hybrids."
    },
    "Corn___Northern_Leaf_Blight": {
        "description": "Long gray-green lesions on leaves.",
        "solution": "Use crop rotation and fungicides."
    },
    "Corn___healthy": {
        "description": "Healthy corn plant.",
        "solution": "No treatment required."
    },

    # 🍇 GRAPE
    "Grape___Black_rot": {
        "description": "Dark spots on leaves and fruit rot.",
        "solution": "Apply fungicide and prune infected areas."
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "Causes leaf discoloration and vine decline.",
        "solution": "Remove infected vines and improve vineyard management."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Brown spots on grape leaves.",
        "solution": "Use fungicide sprays."
    },
    "Grape___healthy": {
        "description": "Healthy grape plant.",
        "solution": "Maintain proper care."
    },

    # 🍑 PEACH
    "Peach___Bacterial_spot": {
        "description": "Small dark lesions on leaves and fruit.",
        "solution": "Use resistant varieties and copper sprays."
    },
    "Peach___healthy": {
        "description": "Healthy plant.",
        "solution": "No treatment needed."
    },

    # 🍓 STRAWBERRY
    "Strawberry___Leaf_scorch": {
        "description": "Leaf edges turn brown and dry.",
        "solution": "Remove infected leaves and apply fungicide."
    },
    "Strawberry___healthy": {
        "description": "Healthy plant.",
        "solution": "Maintain watering and sunlight."
    },

    # 🥔 POTATO
    "Potato___Early_blight": {
        "description": "Dark concentric rings on leaves.",
        "solution": "Use fungicide and proper spacing."
    },
    "Potato___Late_blight": {
        "description": "Rapid leaf decay and rot.",
        "solution": "Apply fungicides and remove infected plants."
    },
    "Potato___healthy": {
        "description": "Healthy plant.",
        "solution": "No treatment required."
    },

    # 🍅 TOMATO
    "Tomato___Bacterial_spot": {
        "description": "Dark spots on leaves and fruits.",
        "solution": "Use copper sprays and avoid overhead irrigation."
    },
    "Tomato___Early_blight": {
        "description": "Dark circular spots with rings.",
        "solution": "Apply fungicide and remove infected leaves."
    },
    "Tomato___Late_blight": {
        "description": "Rapid disease causing leaf rot.",
        "solution": "Use fungicides like metalaxyl."
    },
    "Tomato___Leaf_Mold": {
        "description": "Yellow patches on upper leaf surface.",
        "solution": "Improve ventilation and apply fungicide."
    },
    "Tomato___Septoria_Leaf_Spot": {
        "description": "Small circular spots with gray centers.",
        "solution": "Remove infected leaves and apply fungicide."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Tiny mites causing yellowing and webbing.",
        "solution": "Use insecticidal soap or neem oil."
    },
    "Tomato___Target_Spot": {
        "description": "Dark lesions with concentric rings.",
        "solution": "Apply fungicide and maintain spacing."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Leaves curl and turn yellow due to virus.",
        "solution": "Control whiteflies and remove infected plants."
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Mosaic pattern on leaves.",
        "solution": "Remove infected plants and disinfect tools."
    },
    "Tomato___healthy": {
        "description": "Healthy tomato plant.",
        "solution": "Maintain proper care."
    }
}

def create_dummy_notifications():
    if Notification.query.count() == 0:
        sample = [
            "⚠ Heavy Rain Alert",
            "💧 Irrigation needed",
            "📈 Tomato price increased"
        ]

        for msg in sample:
            db.session.add(Notification(message=msg, type="system"))

        db.session.commit()

def detect_deficiency(n, p, k, ph):
    issues = []

    if n < 40:
        issues.append("Nitrogen deficiency")
    elif n > 120:
        issues.append("Excess Nitrogen")

    if p < 30:
        issues.append("Phosphorus deficiency")
    elif p > 80:
        issues.append("Excess Phosphorus")

    if k < 30:
        issues.append("Potassium deficiency")
    elif k > 80:
        issues.append("Excess Potassium")

    if ph < 5.5:
        issues.append("Soil too acidic")
    elif ph > 7.5:
        issues.append("Soil too alkaline")

    if not issues:
        return ["No major deficiency detected"]

    return issues

def predict_soil_health(n, p, k, temperature, humidity, ph_value, rainfall):
    features = [[n, p, k, temperature, humidity, ph_value, rainfall]]

    soil_idx = soil_model.predict(features)[0]
    soil_type = soil_le.inverse_transform([soil_idx])[0]

    variety_idx = variety_model.predict(features)[0]
    variety = variety_le.inverse_transform([variety_idx])[0]

    deficiency = detect_deficiency(n, p, k, ph_value)

    return soil_type, variety, deficiency

def generate_smart_notifications(user):
    try:
        city = user.city or "Bhubaneswar"
        state = user.state or "Odisha"

        # 🌧 WEATHER DATA
        bttf = BestTimeToFertilize(city_name=city, state_name=state)
        bttf.api_caller()

        if not bttf.is_api_call_success():
            return

        data = bttf.weather_data[0]

        temp = data["Temperature"]
        humidity = data["Relative Humidity"]
        rainfall = data["Rainfall"]

        alerts = []

        # 🌧 WEATHER RULES
        if rainfall > 50:
            alerts.append(f"⚠ Heavy Rain in {city} - Fertilizer wash risk")

        if temp > 35:
            alerts.append(f"🌡 High Temperature in {city} - Irrigation needed")

        if humidity > 80:
            alerts.append(f"🦠 High Humidity in {city} - Fungal disease risk")

        # 📈 SMART MANDI PRICE TRACKING
        mandi_data = get_mandi_prices(state=state, commodity="Tomato")
        print("MANDI DATA:", mandi_data)

        if mandi_data and len(mandi_data) > 0:
            first = mandi_data[0]

            if "modal_price" in first and first["modal_price"]:
                current_price = int(first["modal_price"])

                last_entry = Notification.query.filter_by(type="mandi")\
                    .order_by(Notification.timestamp.desc()).first()

                if last_entry and last_entry.price:
                    diff = current_price - last_entry.price

                    if abs(diff) >= 100:
                        if diff > 0:
                            msg = f"📈 Tomato price increased by ₹{diff} (Now ₹{current_price})"
                        else:
                            msg = f"📉 Tomato price dropped by ₹{abs(diff)} (Now ₹{current_price})"

                        db.session.add(Notification(
                            message=msg,
                            type="mandi",
                            price=current_price
                        ))
                else:
                    db.session.add(Notification(
                        message=f"📊 Tomato current price ₹{current_price}",
                        type="mandi",
                        price=current_price
                    ))
        # 🌧 SAVE WEATHER ALERTS (with time filter)
        from datetime import datetime, timedelta

        for msg in alerts:
            recent = Notification.query.filter(
                Notification.message == msg,
                Notification.timestamp >= datetime.utcnow() - timedelta(hours=6)
            ).first()

            if not recent:
                db.session.add(Notification(message=msg, type="auto"))

        # ✅ COMMIT ONCE (IMPORTANT)
        db.session.commit()

    except Exception as e:
        print("Notification Engine Error:", e)
# ================= IMAGE PREPROCESS ================= #
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ================= FERTILIZER LOGIC ================= #
def get_fertilizer_advice(n, p, k):
    advice = []

    if n < 40:
        advice.append("Apply Urea (Nitrogen deficiency)")
    elif n < 80:
        advice.append("Moderate Nitrogen – balanced fertilizer")
    else:
        advice.append("Nitrogen sufficient")

    if p < 40:
        advice.append("Apply DAP (Phosphorus deficiency)")
    elif p < 80:
        advice.append("Moderate Phosphorus")
    else:
        advice.append("Phosphorus sufficient")

    if k < 40:
        advice.append("Apply MOP (Potassium deficiency)")
    elif k < 80:
        advice.append("Moderate Potassium")
    else:
        advice.append("Potassium sufficient")

    return advice

# ================= CHATBOT ================= #

def detect_intent_and_entities(msg):
    msg = msg.lower()

    intent = "general"
    entities = {}

    if any(x in msg for x in ["fertilizer", "npk", "soil", "nutrient"]):
        intent = "fertilizer"

    elif any(x in msg for x in ["disease", "leaf", "spot", "infection"]):
        intent = "disease"
    
    elif any(x in msg for x in ["price", "mandi", "rate"]):
        intent = "mandi"

    words = msg.split()

    for i, word in enumerate(words):
        if word in ["rice", "wheat", "maize"]:
            entities["crop"] = word

        if word in ["odisha", "bihar", "punjab"]:
            entities["state"] = word

        if i < len(words) - 1 and words[i] == "in":
            entities["city"] = words[i + 1]

    return intent, entities

def get_chatbot_response(user_msg):
    intent, entities = detect_intent_and_entities(user_msg)

    try:
        # ================= FERTILIZER ================= #
        if intent == "fertilizer":

            crop = entities.get("crop")
            state = entities.get("state")
            city = entities.get("city")

            if not crop or not state or not city:
                return {
                    "type": "text",
                    "reply": "Please provide crop, state, and city (e.g., rice in Odisha Bhubaneswar)."
                }

            # 🔗 CALL YOUR EXISTING MODULES
            bttf = BestTimeToFertilize(city_name=city, state_name=state)
            bttf.api_caller()

            if bttf.is_api_call_success() and len(bttf.weather_data) > 0:
                print("API RESPONSE:", bttf.response.json())
                print("WEATHER DATA:", bttf.weather_data)
                return {"type": "text", "reply": "Weather API failed. Try again later."}

            di = bttf.weather_data[0]
            temp = di['Temperature']
            humidity = di['Relative Humidity']
            rainfall = di["Rainfall"]

            est = NPKEstimator()
            est.renameCol()

            n = est.estimator(crop, temp, humidity, rainfall, 'Label_N')
            p = est.estimator(crop, temp, humidity, rainfall, 'Label_P')
            k = est.estimator(crop, temp, humidity, rainfall, 'Label_K')

            advice = get_fertilizer_advice(n, p, k)

            return {
                "type": "text",
                "reply": f"NPK:\nN={n}, P={p}, K={k}\n\n" + "\n".join(advice)
            }

        # ================= DISEASE ================= #
        elif intent == "disease":
            return {
                "type": "redirect",
                "url": "/disease"
            }

         # ================= MANDI ================= #
        elif intent == "mandi":
            state = entities.get("state")
            crop = entities.get("crop")
            if not state or not crop:
                return {
                    "type": "text",
                    "reply": "Please specify crop and state (e.g., rice in Bihar)."
                }
            from mandi_price_module import get_mandi_prices
            data = get_mandi_prices(state=state.capitalize(), commodity=crop.capitalize())
            if not data:
                return {
                    "type": "text",
                    "reply": "No mandi data found."
                }
            reply = "Latest mandi prices:\n"
            for item in data[:3]: 
                reply += f"{item['market']}: ₹{item['modal_price']}\n"
            return {
                "type": "text",
                "reply": reply
            }   

        # ================= GENERAL ================= #
        completion = ai_client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": (
    "You are an agriculture assistant. "
    "Reply in Hindi if user writes in Hindi, otherwise English. "
    "Keep answers simple for farmers."
)},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=150
        )

        return {
            "type": "text",
            "reply": completion.choices[0].message.content.strip()
        }

    except Exception as e:
        print("Chatbot Error:", e)
        return {"type": "text", "reply": "AI service unavailable."}
# ================= LOGIN ================= #
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()

        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials")

    return render_template('login.html')

# ================= SIGNUP ================= #
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':

        if not request.form['email'].endswith("@gmail.com"):
            flash("Only Gmail allowed")
            return redirect(url_for('signup'))

        if User.query.filter_by(email=request.form['email']).first():
            flash("Email exists")
            return redirect(url_for('signup'))

        new_user = User(
            username=request.form['username'],
            email=request.form['email'],
            password=generate_password_hash(request.form['password']),
            phone=request.form['phone']   
        )

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('signup.html')

# ================= LOGOUT ================= #
from flask import session

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()   # 🔥 IMPORTANT FIX
    return redirect(url_for('login'))

# ================= DASHBOARD ================= #
@app.route('/')
@login_required
def dashboard():
    generate_smart_notifications(current_user)
    return render_template('dashboard.html', user=current_user)

@app.route('/fertilizer')
@login_required
def fertilizer():
    return render_template('index.html')

@app.route('/disease')
@login_required
def disease():
    return render_template('disease.html')

@app.route("/soil-health", methods=["GET", "POST"])
@login_required
def soil_health():
    result = None

    if request.method == "POST":
        n = float(request.form["nitrogen"])
        p = float(request.form["phosphorus"])
        k = float(request.form["potassium"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph_value = float(request.form["ph_value"])
        rainfall = float(request.form["rainfall"])

        soil_type, variety, deficiency = predict_soil_health(
            n, p, k, temperature, humidity, ph_value, rainfall
        )

        result = {
            "soil_type": soil_type,
            "variety": variety,
            "deficiency": deficiency
        }

    return render_template("soil_health.html", result=result)

# ================= CHAT ================= #
# ================= CHAT ================= #
@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json() or {}
    user_msg = data.get("message", "")
    result = get_chatbot_response(user_msg)
    return jsonify(result)


# ================= MANDI PRICE ================= #
@app.route('/mandi-price')
@login_required
def mandi_price_page():
    return render_template('mandi_price.html')


@app.route('/api/mandi-price')
@login_required
def mandi_price_api():
    state = request.args.get("state")
    commodity = request.args.get("commodity")
    market = request.args.get("market")

    try:
        data = get_mandi_prices(state, commodity, market)
        return jsonify(data)
    except Exception as e:
        print("Mandi API Error:", e)
        return jsonify([])

# ================= DISEASE PREDICTION ================= #
@app.route('/predict-disease', methods=['POST'])
@login_required
def predict_disease():
    file = request.files.get('image')

    if not file or file.filename == '':
        flash("Please upload an image.")
        return redirect(url_for('disease'))

    upload_dir = os.path.join("static", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    pred = model.predict(img)

    predicted_class = int(np.argmax(pred))
    confidence = float(np.max(pred))

    raw_label = class_labels[predicted_class]
    display_label = raw_label.replace("___", " - ").replace("_", " ")

    # ✅ STEP 1: Initialize as None (will use AI first, then fallback to dict)
    description = None
    solution = None

    # ✅ STEP 2: Try AI as PRIMARY response
    try:
        # Convert raw_label to readable format for prompt
        disease_name = display_label.split(" - ")[-1] if " - " in display_label else display_label
        
        prompt = f"""Provide disease information for: {disease_name}

Please respond in this exact format:
DESCRIPTION: [Brief description of the disease in 1-2 sentences]
SOLUTION: [Practical solution/treatment in 1-2 sentences]"""
        
        completion = ai_client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are an agriculture expert. Provide accurate, practical information for farmers."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )

        ai_text = completion.choices[0].message.content.strip()
        print("AI RAW RESPONSE:", ai_text)  # check terminal

        # Parse AI response - look for DESCRIPTION and SOLUTION
        for line in ai_text.split('\n'):
            line = line.strip()
            if line.upper().startswith("DESCRIPTION:"):
                desc_text = line.split(":", 1)[1].strip()
                if desc_text:
                    description = desc_text.replace("**", "").replace("*", "")
            elif line.upper().startswith("SOLUTION:"):
                sol_text = line.split(":", 1)[1].strip()
                if sol_text:
                    solution = sol_text.replace("**", "").replace("*", "")

        print(f"Parsed - Description: {description}, Solution: {solution}")

    except Exception as e:
        print(f"AI Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ✅ STEP 3: FALLBACK to disease_info dict if AI failed
    if description is None or solution is None:
        print(f"Falling back to disease_info for: {raw_label}")
        info = disease_info.get(raw_label, {})
        if description is None:
            description = info.get("description", "No description available")
        if solution is None:
            solution = info.get("solution", "No solution available")

    return render_template(
        'disease_result.html',
        image=filepath,
        result=display_label,
        confidence=f"{confidence*100:.2f}%",
        description=description,
        solution=solution
    )
# ================= NOTIFICATIONS ================= #

@app.route("/get_notifications")
@login_required
def get_notifications():
    notifications = Notification.query.order_by(Notification.timestamp.desc()).limit(20).all()

    data = []
    unread_count = 0

    for n in notifications:
        if not n.is_read:
            unread_count += 1

        data.append({
            "id": n.id,
            "message": n.message,
            "is_read": n.is_read
        })

    return jsonify({
        "notifications": data,
        "unread_count": unread_count
    })


@app.route("/mark_all_read", methods=["POST"])
@login_required
def mark_all_read():
    Notification.query.filter_by(is_read=False).update({"is_read": True})
    db.session.commit()
    return jsonify({"status": "success"})
# ================= PROCESSING ================= #
@app.route('/processing/', methods=['POST'])
@login_required
def processing():
    global latest_report

    form_data = request.form
    crop = form_data['crop']
    state = form_data['state']
    city = form_data['city']
    current_user.state = state
    current_user.city = city
    db.session.commit()

    call_success = []
    npk_list_dict = []
    popup_data = []
    seven_days = []

    bttf = BestTimeToFertilize(city_name=city, state_name=state)
    bttf.api_caller()

    if bttf.is_api_call_success():
        category, heading, desc = bttf.best_time_fertilize()

        call_success.append(1)
        popup_data.append([category, heading, desc])
        seven_days = bttf.weather_data[:]

        di = bttf.weather_data[0]
        temp = di['Temperature']
        humidity = di['Relative Humidity']
        rainfall = di["Rainfall"]

        est = NPKEstimator()
        est.renameCol()

        npk = {
            'Label_N': est.estimator(crop, temp, humidity, rainfall, 'Label_N'),
            'Label_P': est.estimator(crop, temp, humidity, rainfall, 'Label_P'),
            'Label_K': est.estimator(crop, temp, humidity, rainfall, 'Label_K')
        }

        npk_list_dict.append(npk)

        fertilizer_advice = get_fertilizer_advice(
            npk['Label_N'],
            npk['Label_P'],
            npk['Label_K']
        )

        latest_report = {
            "crop": crop,
            "state": state,
            "city": city,
            "N": npk['Label_N'],
            "P": npk['Label_P'],
            "K": npk['Label_K'],
            "advice": fertilizer_advice
        }

    else:
        flash("Weather API failed")
        fertilizer_advice = []

    return render_template(
        'update.html',
        CALL_SUCCESS=call_success,
        NPK=npk_list_dict,
        FORM_DATA=form_data,
        POPUP_DATA=popup_data,
        SEVEN_DAYS=seven_days,
        FERTILIZER_ADVICE=fertilizer_advice
    )

# ================= PDF ================= #
@app.route('/download-report')
@login_required
def download_report():
    global latest_report

    file_path = "report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Fertilizer Recommendation Report", styles['Title']))
    content.append(Paragraph(f"Crop: {latest_report.get('crop', '-')}", styles['Normal']))
    content.append(Paragraph(f"State: {latest_report.get('state', '-')}", styles['Normal']))
    content.append(Paragraph(f"City: {latest_report.get('city', '-')}", styles['Normal']))

    content.append(Paragraph(" ", styles['Normal']))

    content.append(Paragraph("NPK Values:", styles['Heading2']))
    content.append(Paragraph(f"N: {latest_report.get('N', '-')}", styles['Normal']))
    content.append(Paragraph(f"P: {latest_report.get('P', '-')}", styles['Normal']))
    content.append(Paragraph(f"K: {latest_report.get('K', '-')}", styles['Normal']))

    content.append(Paragraph(" ", styles['Normal']))

    content.append(Paragraph("Fertilizer Advice:", styles['Heading2']))

    for item in latest_report.get("advice", []):
        content.append(Paragraph(f"- {item}", styles['Normal']))

    doc.build(content)

    return send_file(file_path, as_attachment=True)

@app.route('/marketplace')
@login_required
def marketplace():
    search = request.args.get('search', '').strip()
    category = request.args.get('category', '').strip()

    query = Product.query

    # 🔍 Search filter (FIXED)
    if search:
        query = query.filter(Product.name.ilike(f"%{search}%"))

    # 🎯 Category filter (FIXED)
    if category and category != "All":
        query = query.filter(Product.category.ilike(f"%{category}%"))

    products = query.order_by(Product.created_at.desc()).all()

    categories = db.session.query(Product.category).distinct().all()
    categories = [c[0] for c in categories if c[0]]

    return render_template(
        'marketplace.html',
        products=products,
        categories=categories,
        selected_category=category,
        search=search
    )

@app.route('/add-product', methods=['GET', 'POST'])
@login_required
def add_product():
    if request.method == 'POST':
        file = request.files['image']

        filename = file.filename
        filepath = os.path.join('static/uploads/products', filename)
        file.save(filepath)

        product = Product(
            name=request.form['name'],
            description=request.form['description'],
            price=float(request.form['price']),
            quantity=float(request.form['quantity']),
            unit=request.form['unit'],
            category=request.form['category'],
            image=filepath,
            farmer_id=current_user.id
        )

        db.session.add(product)
        db.session.commit()

        return redirect(url_for('marketplace'))

    return render_template('add_product.html')

@app.route('/product/<int:id>')
@login_required
def product_detail(id):
    product = Product.query.get_or_404(id)
    farmer = User.query.get(product.farmer_id)
    return render_template('product_detail.html', product=product, farmer=farmer)

@app.route('/my-listings')
@login_required
def my_listings():
    products = Product.query.filter_by(farmer_id=current_user.id).all()
    return render_template('my_listings.html', products=products)

@app.route('/delete-product/<int:id>')
@login_required
def delete_product(id):
    product = Product.query.get(id)

    if product.farmer_id == current_user.id:
        db.session.delete(product)
        db.session.commit()

    return redirect(url_for('my_listings'))

# ================= RUN ================= #
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        create_dummy_notifications()
    app.run(debug=True)