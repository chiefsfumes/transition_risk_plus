import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
from src.models import Risk, ExternalData
from src.config import NUM_SIMULATIONS, NUM_CLUSTERS, NER_MODEL, SENSITIVITY_VARIABLES, SENSITIVITY_RANGE, TIME_SERIES_HORIZON

def categorize_risks(risks: List[Risk]) -> Dict[str, List[Risk]]:
    categories = {}
    for risk in risks:
        if risk.category not in categories:
            categories[risk.category] = []
        categories[risk.category].append(risk)
    return categories

def analyze_risk_interactions(risks: List[Risk]) -> pd.DataFrame:
    n = len(risks)
    interaction_matrix = np.zeros((n, n))
    for i, risk1 in enumerate(risks):
        for j, risk2 in enumerate(risks):
            if i != j:
                interaction_matrix[i, j] = calculate_interaction_score(risk1, risk2)
    return pd.DataFrame(interaction_matrix, index=[r.id for r in risks], columns=[r.id for r in risks])

def calculate_interaction_score(risk1: Risk, risk2: Risk) -> float:
    # Placeholder for a more sophisticated interaction calculation
    return (risk1.impact * risk2.impact + risk1.likelihood * risk2.likelihood) / 2

def cluster_risks(risks: List[Risk]) -> Dict[int, List[int]]:
    features = [[r.likelihood, r.impact] for r in risks]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    clustered_risks = {i: [] for i in range(NUM_CLUSTERS)}
    for i, cluster in enumerate(clusters):
        clustered_risks[cluster].append(risks[i].id)
    
    return clustered_risks

def analyze_risk_descriptions(risks: List[Risk]) -> Dict[str, List[str]]:
    ner = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple")
    entities = {}
    for risk in risks:
        result = ner(risk.description)
        for entity in result:
            if entity['entity_group'] not in entities:
                entities[entity['entity_group']] = []
            entities[entity['entity_group']].append(entity['word'])
    return entities

def simulate_scenario_impact(risks: List[Risk], external_data: Dict[str, ExternalData], scenario: Dict) -> List[Tuple[Risk, float]]:
    impacts = []
    for risk in risks:
        scenario_impact = (
            risk.impact * (1 + scenario["temp_increase"] / 10) *
            (1 + scenario["carbon_price"] / 1000) *
            (1 - scenario["renewable_energy"] / 2) *
            (1 + scenario["policy_stringency"]) *
            (1 + external_data[max(external_data.keys())].gdp_growth / 100)  # Using the latest year's GDP growth
        )
        impacts.append((risk, round(scenario_impact, 2)))
    return impacts

def monte_carlo_simulation(risks: List[Risk], external_data: Dict[str, ExternalData], scenarios: Dict[str, Dict]) -> Dict[str, Dict[int, List[float]]]:
    results = {scenario: {risk.id: [] for risk in risks} for scenario in scenarios}
    
    latest_external_data = external_data[max(external_data.keys())]
    
    for scenario_name, scenario_params in scenarios.items():
        for _ in range(NUM_SIMULATIONS):
            temp_increase = np.random.normal(scenario_params["temp_increase"], 0.5)
            carbon_price = np.random.normal(scenario_params["carbon_price"], 20)
            renewable_energy = np.random.beta(5 * scenario_params["renewable_energy"], 5 * (1 - scenario_params["renewable_energy"]))
            policy_stringency = np.random.beta(5 * scenario_params["policy_stringency"], 5 * (1 - scenario_params["policy_stringency"]))
            gdp_growth = np.random.normal(latest_external_data.gdp_growth, 1)
            
            for risk in risks:
                simulated_impact = (
                    risk.impact * (1 + temp_increase / 10) * 
                    (1 + carbon_price / 1000) * 
                    (1 - renewable_energy / 2) * 
                    (1 + policy_stringency) *
                    (1 + gdp_growth / 100)
                )
                simulated_likelihood = min(1, max(0, risk.likelihood * np.random.uniform(0.8, 1.2)))
                results[scenario_name][risk.id].append(simulated_impact * simulated_likelihood)
    
    return results

def perform_sensitivity_analysis(risks: List[Risk], scenarios: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    sensitivity_results = {}
    
    for scenario_name, base_scenario in scenarios.items():
        scenario_sensitivity = {}
        
        for variable in SENSITIVITY_VARIABLES:
            high_scenario = base_scenario.copy()
            low_scenario = base_scenario.copy()
            
            high_scenario[variable] *= (1 + SENSITIVITY_RANGE)
            low_scenario[variable] *= (1 - SENSITIVITY_RANGE)
            
            high_impact = sum(impact for _, impact in simulate_scenario_impact(risks, {}, high_scenario))
            low_impact = sum(impact for _, impact in simulate_scenario_impact(risks, {}, low_scenario))
            
            sensitivity = (high_impact - low_impact) / (2 * SENSITIVITY_RANGE * base_scenario[variable])
            scenario_sensitivity[variable] = sensitivity
        
        sensitivity_results[scenario_name] = scenario_sensitivity
    
    return sensitivity_results

def time_series_analysis(risks: List[Risk], external_data: Dict[str, ExternalData]) -> Dict[int, List[float]]:
    time_series_results = {risk.id: [] for risk in risks}
    
    years = sorted(external_data.keys())
    for year in range(years[-1] + 1, years[-1] + TIME_SERIES_HORIZON + 1):
        projected_data = ExternalData(
            year=year,
            gdp_growth=np.mean([external_data[y].gdp_growth for y in years[-5:]]),
            population=external_data[years[-1]].population * (1 + 0.01),  # Assuming 1% annual population growth
            energy_demand=external_data[years[-1]].energy_demand * (1 + 0.02)  # Assuming 2% annual energy demand growth
        )
        
        for risk in risks:
            projected_impact = (
                risk.impact * 
                (1 + projected_data.gdp_growth / 100) * 
                (1 + np.log(projected_data.population / external_data[years[-1]].population)) *
                (1 + np.log(projected_data.energy_demand / external_data[years[-1]].energy_demand))
            )
            time_series_results[risk.id].append(projected_impact)
    
    return time_series_results