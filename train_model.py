import csv
import pickle
import os
import random
import heapq
from math import sqrt

DATA_PATH = "data/autos.csv"
MODEL_DIR = "model"
MODEL_PATH = "model/car_price_model.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)

MAX_VALID_ROWS = 1000
MAX_TEST_ROWS = 100


def clean_text(value):
    if value is None:
        return "unknown"

    value = str(value).strip().lower()

    if value == "" or value == "nan":
        return "unknown"

    return value


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except:
        return default


def detect_delimiter(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        sample = file.read(2048)

    if sample.count(";") > sample.count(","):
        return ";"
    return ","


def load_data(path):
    cars = []
    delimiter = detect_delimiter(path)

    print("Delimiter détecté :", delimiter)
    print("Lecture du fichier autos.csv ...")

    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)

        for i, row in enumerate(reader):
            if i % 10000 == 0:
                print("Lignes lues :", i, "| voitures valides :", len(cars))

            price = safe_int(row.get("price"), 0)
            year = safe_int(row.get("yearOfRegistration"), 0)
            kilometer = safe_int(row.get("kilometer"), 0)
            power = safe_int(row.get("powerPS"), 0)

            brand = clean_text(row.get("brand"))
            model = clean_text(row.get("model"))
            fuel_type = clean_text(row.get("fuelType"))
            gearbox = clean_text(row.get("gearbox"))
            vehicle_type = clean_text(row.get("vehicleType"))
            damage = clean_text(row.get("notRepairedDamage"))

            if price < 500 or price > 100000:
                continue

            if year < 1980 or year > 2026:
                continue

            if kilometer <= 0:
                continue

            if power <= 0 or power > 1000:
                continue

            if brand == "unknown" or model == "unknown":
                continue

            car = {
                "brand": brand,
                "model": model,
                "year": year,
                "mileage": kilometer,
                "fuel_type": fuel_type,
                "transmission": gearbox,
                "engine_power": power,
                "vehicle_type": vehicle_type,
                "damage": damage,
                "price": float(price)
            }

            cars.append(car)

            if len(cars) >= MAX_VALID_ROWS:
                break

    return cars


def get_numeric_ranges(cars):
    numeric_columns = ["year", "mileage", "engine_power"]
    ranges = {}

    for col in numeric_columns:
        values = [car[col] for car in cars]
        value_range = max(values) - min(values)

        if value_range == 0:
            value_range = 1

        ranges[col] = value_range

    return ranges


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


def predict_price(car, training_data, ranges, k=5):
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


def price_to_class(price):
    if price < 5000:
        return "Bas"
    elif price <= 15000:
        return "Moyen"
    else:
        return "Élevé"


cars = load_data(DATA_PATH)

if len(cars) < 50:
    raise Exception("Dataset trop petit après nettoyage. Vérifie autos.csv.")

print("\nNombre final de voitures utilisées :", len(cars))

random.seed(42)
random.shuffle(cars)

split_index = int(len(cars) * 0.8)
train_data = cars[:split_index]
test_data = cars[split_index:]
test_data = test_data[:MAX_TEST_ROWS]

ranges = get_numeric_ranges(train_data)

errors = []
correct_classes = 0

confusion_matrix = {
    "Bas": {"Bas": 0, "Moyen": 0, "Élevé": 0},
    "Moyen": {"Bas": 0, "Moyen": 0, "Élevé": 0},
    "Élevé": {"Bas": 0, "Moyen": 0, "Élevé": 0}
}

print("\nÉvaluation du modèle ...")

for index, car in enumerate(test_data):
    if index % 50 == 0:
        print("Tests effectués :", index, "/", len(test_data))

    real_price = car["price"]
    predicted_price = predict_price(car, train_data, ranges, k=5)

    errors.append(abs(real_price - predicted_price))

    real_class = price_to_class(real_price)
    pred_class = price_to_class(predicted_price)

    if real_class == pred_class:
        correct_classes += 1

    confusion_matrix[real_class][pred_class] += 1

mae = sum(errors) / len(errors)
accuracy_class = correct_classes / len(test_data) * 100

print("\n✅ Modèle entraîné avec succès")
print("Algorithme : Manual KNN Regressor")
print("Train :", len(train_data))
print("Test :", len(test_data))
print("MAE :", round(mae, 2), "€")
print("Accuracy classes :", round(accuracy_class, 2), "%")

print("\nMatrice de confusion :")
print("Réel \\ Prédit | Bas | Moyen | Élevé")
print("------------------------------------")

for real in ["Bas", "Moyen", "Élevé"]:
    print(
        real,
        "|",
        confusion_matrix[real]["Bas"],
        "|",
        confusion_matrix[real]["Moyen"],
        "|",
        confusion_matrix[real]["Élevé"]
    )

model = {
    "algorithm": "Manual KNN Regressor",
    "training_data": train_data,
    "ranges": ranges,
    "k": 5
}

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("\n✅ Modèle sauvegardé dans :", MODEL_PATH)