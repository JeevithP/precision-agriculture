def irrigation_recommendation(soil_moisture, rainfall):
    if soil_moisture < 30 and rainfall < 50:
        return "High irrigation required"
    elif soil_moisture < 50:
        return "Moderate irrigation required"
    else:
        return "Low irrigation required"

def fertilizer_recommendation(N, P, K):
    if N < 40:
        return "Add Nitrogen fertilizer"
    elif P < 40:
        return "Add Phosphorus fertilizer"
    elif K < 40:
        return "Add Potassium fertilizer"
    else:
        return "NPK levels are balanced"
