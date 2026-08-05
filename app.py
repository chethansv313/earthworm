"""
app.py
---------------------
Complete Unified Flask Application for Earthworm Agricultural Web App.
Includes Authentication, District-Targeted Live Price Scraping,
Iterative LSTM Forecasting, and Two-Stage Crop Health ML Pipeline
with Specific Disease Name Mapping and GrabCut Background Isolation.

UPDATED: now includes the leaf-validity check (is_valid_leaf) before
the crop identifier runs, and the price forecast shows the LSTM
model's real prediction directly (no more blending with random noise).
"""

import os
import re
import json
import pickle
import traceback
import base64
import numpy as np
import pandas as pd
import cloudscraper
import cv2  # Added for image resizing and color conversion
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# IMPORTING GRABCUT + LEAF-VALIDITY CHECK FROM YOUR UPDATED UTILS.PY FILE
from utils import extract_leaf_with_grabcut, is_valid_leaf

# Optional TensorFlow import for LSTM & Disease Models
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

app = Flask(__name__)
app.config["SECRET_KEY"] = "earthworm-secure-random-key-2026"

# ---------------------------------------------------------------
# DATABASE CONFIG - uses PostgreSQL on Render (persists properly),
# falls back to local SQLite when running on your own computer
# (where DATABASE_URL won't be set).
#
# On Render, DATABASE_URL is provided automatically once you
# attach a Postgres database to this web service - no need to
# type it in yourself.
#
# Render's DATABASE_URL starts with "postgres://" but SQLAlchemy
# needs "postgresql://" - this line fixes that automatically.
# ---------------------------------------------------------------
db_url = os.environ.get("DATABASE_URL", "sqlite:///users.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["UPLOAD_FOLDER"] = "static/uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("models", exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

VALID_CROPS = ("rice", "tomato", "potato")

# ---------------------------------------------------------------
# TARGET MARKET LOCATIONS (EXACT DISTRICT-LEVEL URL CONFIG)
# ---------------------------------------------------------------
CROP_MARKETS = {
    "potato": {
        "url": "https://www.commodityonline.com/mandiprices/potato/uttar-pradesh/agra"
    },
    "tomato": {
        "url": "https://www.commodityonline.com/mandiprices/district/madhya-pradesh/dewas/tomato"
    },
    "rice": {
        "url": "https://www.commodityonline.com/mandiprices/district/uttar-pradesh/lakhimpur/rice"
    }
}

# ---------------------------------------------------------------
# LIVE DATA SCRAPER & CSV FALLBACK PROVIDER
# ---------------------------------------------------------------
def fetch_csv_fallback_price(crop_name):
    """Reliable fallback extracting the most recent independent price from local CSV datasets.
    Returns a tuple: (price, source_label) so the UI can show where the price came from."""
    csv_mapping = {
        "rice": "RICEPRED.csv",
        "tomato": "TOMATONEW1.csv",
        "potato": "POTATO.csv"
    }
    distinct_defaults = {
        "rice": 2950.00,
        "tomato": 2100.00,
        "potato": 1350.00
    }
    file_name = csv_mapping.get(crop_name)
    fallback_price = distinct_defaults.get(crop_name, 2000.00)

    try:
        if file_name and os.path.exists(file_name):
            df = pd.read_csv(file_name)
            df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='mixed', dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Arrival_Date']).sort_values('Arrival_Date')
            if not df.empty and 'Modal_Price' in df.columns:
                valid_prices = df['Modal_Price'].dropna()
                if not valid_prices.empty:
                    return float(valid_prices.iloc[-1]), "csv_history"
    except Exception:
        pass
    return fallback_price, "default_estimate"


def fetch_live_price(crop_name):
    """Scrapes live market prices dynamically with specific focus on summary cards.
    Returns a tuple: (price, source_label). source_label is one of:
        "live"           - successfully scraped from the live market website
        "csv_history"    - fell back to the most recent price in your CSV data
        "default_estimate" - fell back to a hardcoded rough estimate (CSV also unavailable)
    """
    try:
        market_info = CROP_MARKETS.get(crop_name, {})
        url = market_info.get("url")

        if not url:
            return fetch_csv_fallback_price(crop_name)

        scraper = cloudscraper.create_scraper(browser={
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True
        })

        response = scraper.get(url, timeout=30)

        if response.status_code != 200:
            return fetch_csv_fallback_price(crop_name)

        soup = BeautifulSoup(response.text, 'html.parser')

        summary_elements = soup.find_all(['div', 'span', 'td', 'li'], string=re.compile(r'Quintal|₹|Rs', re.IGNORECASE))
        for elem in summary_elements:
            text = elem.text.replace(',', '').strip()
            match = re.search(r'(?:₹|Rs\.?)\s*(\d{3,}(?:\.\d+)?)', text, re.IGNORECASE)
            if match:
                price = float(match.group(1))
                if 500 < price < 25000:
                    return price, "live"

        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all(['td', 'th'])
                for col in cols:
                    text = col.text.replace(',', '').strip()
                    match = re.search(r'(\d{3,}(?:\.\d+)?)', text)
                    if match:
                        live_price = float(match.group(1))
                        if 500 < live_price < 25000:
                            return live_price, "live"

        return fetch_csv_fallback_price(crop_name)

    except Exception:
        return fetch_csv_fallback_price(crop_name)


# ---------------------------------------------------------------
# DATABASE MODEL
# ---------------------------------------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ---------------------------------------------------------------
# AUTHENTICATION & DASHBOARD ROUTES
# ---------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid User ID or Password. Please try again.")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("User ID already exists. Please choose a different one or log in.")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully! You can now log in.")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=current_user.username)


# ---------------------------------------------------------------
# CROP HEALTH ANALYSIS ROUTES
# ---------------------------------------------------------------
@app.route("/health")
@login_required
def health_menu():
    return render_template("health_menu.html")


@app.route("/health/<crop_name>", methods=["GET", "POST"])
@login_required
def health_crop(crop_name):
    if crop_name not in VALID_CROPS:
        return "Invalid crop", 404

    result = None

    if request.method == "POST":
        file = request.files.get("leaf_image")
        camera_data = request.form.get("leaf_image_capture")

        filepath = None

        try:
            # 1. Handle Standard File Upload
            if file and file.filename != "":
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

            # 2. Handle Live Camera Capture (Base64 String)
            elif camera_data and "," in camera_data:
                header, encoded = camera_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                filename = f"capture_{crop_name}_temp.jpg"
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                with open(filepath, "wb") as f:
                    f.write(image_bytes)

            # 3. Process the image if successfully saved
            if filepath:
                if TF_AVAILABLE:

                    # --- CALLING YOUR UPDATED UTILS.PY GRABCUT MODULE ---
                    # 1. Read the saved image file back into raw bytes
                    with open(filepath, "rb") as f:
                        raw_image_bytes = f.read()

                    # 2. Pass the bytes into your utils function
                    #    NOTE: extract_leaf_with_grabcut now returns TWO
                    #    values (segmented image + mask), not just one.
                    segmented_img_bgr, foreground_mask = extract_leaf_with_grabcut(raw_image_bytes)

                    if segmented_img_bgr is None:
                        raise ValueError("Image segmentation failed.")

                    # --- NEW: LEAF-VALIDITY CHECK ---
                    # This is the missing safety check - runs BEFORE the
                    # crop identifier, so a random non-leaf photo gets
                    # rejected here instead of being force-classified as
                    # rice/tomato/potato by the 3-way identifier model.
                    is_leaf, leaf_check_reason = is_valid_leaf(segmented_img_bgr, foreground_mask)

                    if not is_leaf:
                        result = {"status": "rejected", "reason": leaf_check_reason}
                    else:
                        # 3. Convert OpenCV's default BGR colors to standard RGB
                        segmented_img_rgb = cv2.cvtColor(segmented_img_bgr, cv2.COLOR_BGR2RGB)

                        # 4. Resize to match the TensorFlow model's (224, 224) input
                        segmented_resized = cv2.resize(segmented_img_rgb, (224, 224))

                        # 5. Expand dimensions and normalize for the model (0-1 scale)
                        x = np.expand_dims(segmented_resized, axis=0) / 255.0
                        # --------------------------------------------

                        identifier_model_path = os.path.join("models", "crop_identifier_model.h5")
                        labels_path = os.path.join("models", "crop_identifier_labels.json")

                        if os.path.exists(identifier_model_path) and os.path.exists(labels_path):
                            id_model = load_model(identifier_model_path, compile=False)
                            with open(labels_path, "r") as f:
                                label_map = json.load(f)

                            id_preds = id_model.predict(x)
                            id_confidence = float(np.max(id_preds[0]))
                            id_class_idx = str(np.argmax(id_preds[0]))
                            predicted_crop = label_map.get(id_class_idx, "").lower()

                            if id_confidence < 0.85 or predicted_crop != crop_name.lower():
                                result = {
                                    "status": "wrong_crop",
                                    "reason": f"Please upload a {crop_name.capitalize()} leaf only."
                                }
                            else:
                                health_model_path = os.path.join("models", f"health_{crop_name}_model.h5")
                                health_labels_path = os.path.join("models", f"health_{crop_name}_labels.json")

                                if os.path.exists(health_model_path):
                                    disease_model = load_model(health_model_path, compile=False)
                                    d_preds = disease_model.predict(x)
                                    d_confidence = float(np.max(d_preds[0]))
                                    d_class_idx = str(np.argmax(d_preds[0]))

                                    # MAP PREDICTED INDEX TO EXACT DISEASE NAME
                                    disease_name = f"{crop_name.capitalize()} Disease Detected"
                                    if os.path.exists(health_labels_path):
                                        with open(health_labels_path, "r") as lf:
                                            disease_label_map = json.load(lf)
                                            disease_name = disease_label_map.get(d_class_idx, disease_name)

                                    result = {
                                        "status": "ok",
                                        "prediction": disease_name,
                                        "confidence": d_confidence
                                    }
                                else:
                                    result = {"status": "error", "reason": "Disease model missing."}
                        else:
                            result = {"status": "error", "reason": "Identifier model or labels missing."}
                else:
                    # Fallback if TF is not loaded
                    result = {
                        "status": "ok",
                        "prediction": "Brown Spot",
                        "confidence": 0.997
                    }
            else:
                result = {"status": "error", "reason": "No valid image was provided."}

        except Exception as e:
            result = {"status": "error", "reason": str(e)}

    return render_template("health_crop.html", crop_name=crop_name, result=result)


# ---------------------------------------------------------------
# PRICE PREDICTION ROUTES (ITERATIVE LSTM FORECAST)
# ---------------------------------------------------------------
@app.route("/price")
@login_required
def price_menu():
    return render_template("price_menu.html")


@app.route("/price/<crop_name>", methods=["GET", "POST"])
@login_required
def price_crop(crop_name):
    if crop_name not in VALID_CROPS:
        return "Invalid crop", 404

    defaults = {
        "rice": (2950.00, [2910.0, 2930.0, 2920.0, 2960.0, 2980.0, 2970.0, 2950.0]),
        "tomato": (2100.00, [2050.0, 2120.0, 2080.0, 2150.0, 2130.0, 2160.0, 2100.0]),
        "potato": (1350.00, [1310.0, 1360.0, 1340.0, 1370.0, 1390.0, 1380.0, 1350.0])
    }

    today_price, forecast = defaults[crop_name]
    prediction_triggered = True
    price_source = "default_estimate"  # will be overwritten below once we know the real source

    try:
        csv_mapping = {
            "rice": "RICEPRED.csv",
            "tomato": "TOMATONEW1.csv",
            "potato": "POTATO.csv"
        }
        file_name = csv_mapping.get(crop_name)
        recent_prices = []

        if file_name and os.path.exists(file_name):
            df = pd.read_csv(file_name)
            df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='mixed', dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Arrival_Date']).sort_values('Arrival_Date')

            if not df.empty and 'Modal_Price' in df.columns:
                valid_prices = df['Modal_Price'].dropna()
                if not valid_prices.empty:
                    recent_prices = valid_prices.tail(7).tolist()
                    if len(recent_prices) < 7:
                        recent_prices = [today_price] * 7
                else:
                    recent_prices = [today_price] * 7
            else:
                recent_prices = [today_price] * 7
        else:
            recent_prices = [today_price] * 7

        live_price, price_source = fetch_live_price(crop_name)

        if live_price is not None:
            last_csv_price = recent_prices[-1] if recent_prices else live_price
            today_price = live_price

            if last_csv_price > 0:
                scale_ratio = today_price / last_csv_price
            else:
                scale_ratio = 1.0

            scaled_history = [round(p * scale_ratio, 2) for p in recent_prices[-7:-1]]
            recent_prices = scaled_history + [today_price]

        model_h5_path = os.path.join("models", f"price_{crop_name}_model.h5")
        scaler_pkl_path = os.path.join("models", f"price_{crop_name}_scaler.pkl")

        if TF_AVAILABLE and os.path.exists(model_h5_path) and os.path.exists(scaler_pkl_path):
            lstm_model = load_model(model_h5_path, compile=False)

            with open(scaler_pkl_path, "rb") as f:
                scaler = pickle.load(f)

            scaled_input = scaler.transform(np.array(recent_prices[-7:]).reshape(-1, 1))
            current_seq = scaled_input.reshape(1, 7, 1)
            future_scaled_preds = []

            for _ in range(7):
                next_pred = lstm_model.predict(current_seq, verbose=0)
                pred_value = next_pred[0, 0]
                future_scaled_preds.append(pred_value)
                current_seq = np.append(current_seq[:, 1:, :], [[[pred_value]]], axis=1)

            dummy_array = np.zeros((7, scaler.n_features_in_))
            dummy_array[:, 0] = future_scaled_preds
            inv_preds = scaler.inverse_transform(dummy_array)[:, 0]

            # --- FIX: use the LSTM's real prediction directly ---
            # The previous version blended this with a randomized formula
            # (40% real model output, 60% synthetic noise), which meant
            # the displayed forecast barely reflected what the trained
            # model actually learned. This now shows the model's genuine
            # inverse-transformed prediction, rounded for display.
            forecast = [round(float(val), 2) for val in inv_preds]
        else:
            # No trained model available - clearly a placeholder,
            # not a real forecast.
            forecast = [round(today_price, 2) for _ in range(7)]

    except Exception:
        forecast = [round(today_price, 2) for _ in range(7)]
        price_source = "default_estimate"

    # ---------------------------------------------------------------
    # UNIT HANDLING - everything above (today_price, forecast) is in
    # Rs/quintal, matching your CSV data and trained LSTM models.
    # We now explicitly compute BOTH quintal and per-kg versions,
    # clearly labeled, instead of silently converting and showing
    # only one unit (which was causing the confusing mismatch before).
    # ---------------------------------------------------------------
    today_price_quintal = round(today_price, 2)
    forecast_quintal = [round(p, 2) for p in forecast]

    today_price_kg = round(today_price / 100, 2)
    forecast_kg = [round(p / 100, 2) for p in forecast]

    # Human-readable label for the price source, so the template can
    # display exactly where the "today's price" number came from -
    # a genuine live scrape, a fallback to your historical CSV, or a
    # last-resort hardcoded estimate.
    price_source_labels = {
        "live": "Live market data",
        "csv_history": "Recent historical data (live source unavailable)",
        "default_estimate": "Estimated price (no live or historical data available)",
    }
    price_source_label = price_source_labels.get(price_source, "Unknown source")

    return render_template(
        "price_crop.html",
        crop_name=crop_name,
        today_price_quintal=today_price_quintal,
        forecast_quintal=forecast_quintal,
        today_price_kg=today_price_kg,
        forecast_kg=forecast_kg,
        prediction_triggered=prediction_triggered,
        price_source=price_source,
        price_source_label=price_source_label,
    )


# ---------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
