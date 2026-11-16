from flask import Flask, request, jsonify, render_template
import pickle
import os
import heapq
from math import sqrt

app = Flask(__name__)

MODEL_PATH = "model/car_price_model.pkl"
EUR_TO_TND = 3.35

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Le fichier model/car_price_model.pkl n'existe pas. "
        "Exécute d'abord : python train_model.py"
    )

with open(MODEL_PATH, "rb") as file:
    model_data = pickle.load(file)

training_data = model_data["training_data"]
ranges = model_data["ranges"]
k = model_data["k"]


def clean_text(value):
    if value is None:
        return "unknown"

    value = str(value).strip().lower()

    if value == "":
        return "unknown"

    return value


def distance(car1, car2, ranges):
    d = 0

    d += ((car1["year"] - car2["year"]) / ranges["year"]) ** 2
    d += ((car1["mileage"] - car2["mileage"]) / ranges["mileage"]) ** 2
    d += ((car1["engine_power"] - car2["engine_power"]) / ranges["engine_power"]) ** 2

    d += 0 if car1["brand"] == car2["brand"] else 1
    d += 0 if car1["model"] == car2["model"] else 1
    d += 0 if car1["fuel_type"] == car2["fuel_type"] else 0.5
    d += 0 if car1["transmission"] == car2["transmission"] else 0.5
    d += 0 if car1["vehicle_type"] == car2["vehicle_type"] else 0.5
    d += 0 if car1["damage"] == car2["damage"] else 0.5

    return sqrt(d)


def predict_price(car):
    distances = []

    for train_car in training_data:
        dist = distance(car, train_car, ranges)
        distances.append((dist, train_car["price"]))

    nearest = heapq.nsmallest(k, distances, key=lambda x: x[0])

    total_weight = 0
    weighted_sum = 0

    for dist, price in nearest:
        weight = 1 / (dist + 0.0001)
        weighted_sum += price * weight
        total_weight += weight

    return weighted_sum / total_weight


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        car = {
            "brand": clean_text(data["brand"]),
            "model": clean_text(data["model"]),
            "year": int(data["year"]),
            "mileage": int(data["mileage"]),
            "fuel_type": clean_text(data["fuel_type"]),
            "transmission": clean_text(data["transmission"]),
            "engine_power": int(data["engine_power"]),
            "vehicle_type": clean_text(data["vehicle_type"]),
            "damage": clean_text(data["damage"])
        }

        prediction_eur = predict_price(car)
        prediction_tnd = prediction_eur * EUR_TO_TND

        return jsonify({
            "success": True,
            "predicted_price_eur": round(prediction_eur, 0),
            "predicted_price_tnd": round(prediction_tnd, 0),
            "conversion_rate": EUR_TO_TND
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)