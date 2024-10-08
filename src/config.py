import os
import logging

# Scenario definitions
SCENARIOS = {
    "Net Zero 2050": {"temp_increase": 1.5, "carbon_price": 250, "renewable_energy": 0.75, "policy_stringency": 0.9},
    "Delayed Transition": {"temp_increase": 2.5, "carbon_price": 125, "renewable_energy": 0.55, "policy_stringency": 0.6},
    "Current Policies": {"temp_increase": 3.5, "carbon_price": 35, "renewable_energy": 0.35, "policy_stringency": 0.2}
}

# Monte Carlo simulation parameters
NUM_SIMULATIONS = 10000

# Clustering parameters
NUM_CLUSTERS = 3

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# NLP model
NER_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"

# Visualization settings
VIZ_DPI = 300
HEATMAP_CMAP = 'YlOrRd'

# Time series analysis parameters
TIME_SERIES_HORIZON = 10  # years

# Sensitivity analysis parameters
SENSITIVITY_VARIABLES = ['temp_increase', 'carbon_price', 'renewable_energy', 'policy_stringency']
SENSITIVITY_RANGE = 0.2  # +/- 20%

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.path.join(OUTPUT_DIR, 'risk_assessment.log')
    )