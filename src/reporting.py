import json
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
import os
from src.models import Risk
from src.config import OUTPUT_DIR

def generate_report(risks: List[Risk], categorized_risks: Dict[str, List[Risk]], 
                    interaction_matrix: pd.DataFrame, scenario_impacts: Dict[str, List[Tuple[Risk, float]]],
                    simulation_results: Dict[str, Dict[int, List[float]]], clustered_risks: Dict[int, List[int]],
                    risk_entities: Dict[str, List[str]], sensitivity_results: Dict[str, Dict[str, float]],
                    time_series_results: Dict[int, List[float]], mitigation_strategies: Dict[int, List[str]]) -> str:
    report = {
        "total_risks": len(risks),
        "risk_categories": {category: len(risks) for category, risks in categorized_risks.items()},
        "high_impact_risks": [risk.to_dict() for risk in risks if risk.impact > 0.7],
        "risk_interactions": interaction_matrix.to_dict(),
        "scenario_analysis": {
            scenario: [
                {"risk_id": risk.id, "impact": impact} 
                for risk, impact in sorted(impacts, key=lambda x: x[1], reverse=True)[:3]
            ] for scenario, impacts in scenario_impacts.items()
        },
        "monte_carlo_results": {
            scenario: {
                risk_id: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "5th_percentile": np.percentile(values, 5),
                    "95th_percentile": np.percentile(values, 95)
                } for risk_id, values in scenario_results.items()
            } for scenario, scenario_results in simulation_results.items()
        },
        "risk_clusters": clustered_risks,
        "risk_entities": risk_entities,
        "sensitivity_analysis": sensitivity_results,
        "time_series_projection": {risk_id: projections for risk_id, projections in time_series_results.items()},
        "mitigation_strategies": mitigation_strategies
    }
    
    report_json = json.dumps(report, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'climate_risk_report.json'), 'w') as f:
        f.write(report_json)
    
    return report_json