import pandas as pd
from typing import List, Dict
from src.models import Risk, ExternalData

def load_risk_data(file_path: str) -> List[Risk]:
    df = pd.read_csv(file_path)
    return [Risk(row['id'], row['description'], row['category'], row['likelihood'], row['impact']) for _, row in df.iterrows()]

def load_external_data(file_path: str) -> Dict[str, ExternalData]:
    df = pd.read_csv(file_path)
    return {row['year']: ExternalData(row['year'], row['gdp_growth'], row['population'], row['energy_demand']) 
            for _, row in df.iterrows()}