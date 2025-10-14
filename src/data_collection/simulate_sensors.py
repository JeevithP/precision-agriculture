import random

def simulate_sensor_data():
    return {
        "N": random.randint(10, 140),
        "P": random.randint(5, 145),
        "K": random.randint(5, 205),
        "temperature": round(random.uniform(15, 35), 2),
        "humidity": round(random.uniform(30, 90), 2),
        "ph": round(random.uniform(4, 9), 2),
        "rainfall": round(random.uniform(20, 300), 2)
    }
