import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd
import os
from src.models import Risk
from src.config import OUTPUT_DIR, VIZ_DPI, HEATMAP_CMAP, TIME_SERIES_HORIZON

def generate_visualizations(risks: List[Risk], interaction_matrix: pd.DataFrame, 
                            simulation_results: Dict[str, Dict[int, List[float]]],
                            sensitivity_results: Dict[str, Dict[str, float]],
                            time_series_results: Dict[int, List[float]]):
    # Risk Matrix
    plt.figure(figsize=(10, 8))
    for risk in risks:
        plt.scatter(risk.likelihood, risk.impact, s=100)
        plt.annotate(risk.id, (risk.likelihood, risk.impact))
    plt.xlabel('Likelihood')
    plt.ylabel('Impact')
    plt.title('Risk Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'risk_matrix.png'), dpi=VIZ_DPI)
    plt.close()

    # Interaction Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(interaction_matrix, annot=True, cmap=HEATMAP_CMAP)
    plt.title('Risk Interaction Heatmap')
    plt.savefig(os.path.join(OUTPUT_DIR, 'interaction_heatmap.png'), dpi=VIZ_DPI)
    plt.close()

    # Monte Carlo Simulation Results
    plt.figure(figsize=(15, 10))
    for scenario, results in simulation_results.items():
        for risk_id, values in results.items():
            sns.kdeplot(values, label=f'Risk {risk_id} - {scenario}')
    plt.xlabel('Risk Impact')
    plt.ylabel('Density')
    plt.title('Monte Carlo Simulation Results')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'monte_carlo_results.png'), dpi=VIZ_DPI)
    plt.close()

    # Sensitivity Analysis Heatmap
    plt.figure(figsize=(12, 8))
    sensitivity_df = pd.DataFrame(sensitivity_results).T
    sns.heatmap(sensitivity_df, annot=True, cmap='coolwarm', center=0)
    plt.title('Sensitivity Analysis Heatmap')
    plt.savefig(os.path.join(OUTPUT_DIR, 'sensitivity_analysis_heatmap.png'), dpi=VIZ_DPI)
    plt.close()

    # Time Series Projection
    plt.figure(figsize=(15, 10))
    for risk_id, projections in time_series_results.items():
        plt.plot(range(1, TIME_SERIES_HORIZON + 1), projections, label=f'Risk {risk_id}')
    plt.xlabel('Years into the future')
    plt.ylabel('Projected Impact')
    plt.title('Time Series Projection of Risk Impacts')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_series_projection.png'), dpi=VIZ_DPI)
    plt.close()