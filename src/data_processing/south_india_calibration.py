import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = "data/south_india_soil.csv"

def load_soil_data():
    df = pd.read_csv(DATA_PATH)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    return df


def get_district_scaler(district):
    df = load_soil_data()

    # ---- EXACT COLUMN MAPPING (BASED ON YOUR DATASET) ----
    required_columns = {
        "n": "nitrogen value",
        "p": "phosphorous value",
        "k": "potassium value",
        "ph": "ph"
    }

    # Verify columns exist
    for key, col in required_columns.items():
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found. Found columns: {df.columns.tolist()}"
            )

    # Normalize district column
    df["district"] = df["district"].astype(str).str.strip().str.lower()
    district = district.strip().lower()

    district_df = df[df["district"] == district]

    # Fallback: use all South India data
    if district_df.empty:
        district_df = df

    scaler = MinMaxScaler()
    scaler.fit(
        district_df[
            [
                required_columns["n"],
                required_columns["p"],
                required_columns["k"],
                required_columns["ph"]
            ]
        ]
    )

    return scaler


def calibrate_input(input_data, district):
    scaler = get_district_scaler(district)

    soil_values = [[
        input_data["N"],
        input_data["P"],
        input_data["K"],
        input_data["ph"]
    ]]

    scaled = scaler.transform(soil_values)

    input_data["N"] = scaled[0][0]
    input_data["P"] = scaled[0][1]
    input_data["K"] = scaled[0][2]
    input_data["ph"] = scaled[0][3]

    return input_data
