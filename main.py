import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Load the saved pipeline
pipeline = joblib.load("models/house_price_model.pkl")

class HouseData(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str


def format_indian_currency(amount):
    """Convert number to Indian currency format with commas (e.g. ₹34,70,431.28)"""
    amount = round(float(amount), 2)
    
    str_amount = f"{amount:.2f}"
    integer_part, decimal_part = str_amount.split(".")
    
    is_negative = integer_part.startswith("-")
    if is_negative:
        integer_part = integer_part[1:]
    
    # Indian grouping: last 3 digits together, then groups of 2
    if len(integer_part) > 3:
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]
        parts = []
        while len(remaining) > 2:
            parts.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            parts.insert(0, remaining)
        formatted_integer = ",".join(parts) + "," + last_three
    else:
        formatted_integer = integer_part
    
    result = f"₹{formatted_integer}.{decimal_part}"
    return f"-{result}" if is_negative else result


# Route to serve the HTML interface
@app.get("/")
def read_root():
    return FileResponse("templates/index.html")

# Route to process predictions
@app.post("/predict")
def predict(data: HouseData):
    # Convert input to dictionary
    input_dict = data.dict()
    
    # Binary mapping (Matches training logic)
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        input_dict[col] = 1 if input_dict[col].lower() == 'yes' else 0
        
    # Convert to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Feature Engineering (Must match the exact logic from train.py)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['area_per_room'] = df['area'] / df['total_rooms']
    
    # Pipeline handles preprocessing and prediction
    log_prediction = pipeline.predict(df)
    
    # Reverse log transformation
    price = np.expm1(log_prediction[0])
    
    return {
        "predicted_price": round(float(price), 2),
        "predicted_price_formatted": format_indian_currency(price)
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)