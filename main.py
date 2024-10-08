import os
import logging
from src.data_loader import load_risk_data, load_external_data
from src.risk_analysis import (
    categorize_risks,
    analyze_risk_interactions,
    simulate_scenario_impact,
    monte_carlo_simulation,
    cluster_risks,
    analyze_risk_descriptions,
    perform_sensitivity_analysis,
    time_series_analysis
)
from src.visualization import generate_visualizations
from src.reporting import generate_report
from src.mitigation import suggest_mitigation_strategies
from src.config import SCENARIOS, OUTPUT_DIR, setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Advanced Climate Risk Assessment Tool")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Load and process data
        risks = load_risk_data('data/risk_data.csv')
        external_data = load_external_data('data/external_data.csv')
        
        categorized_risks = categorize_risks(risks)
        interaction_matrix = analyze_risk_interactions(risks)
        
        scenario_impacts = {
            scenario: simulate_scenario_impact(risks, external_data, params)
            for scenario, params in SCENARIOS.items()
        }
        
        simulation_results = monte_carlo_simulation(risks, external_data, SCENARIOS)
        clustered_risks = cluster_risks(risks)
        risk_entities = analyze_risk_descriptions(risks)
        
        sensitivity_results = perform_sensitivity_analysis(risks, SCENARIOS)
        time_series_results = time_series_analysis(risks, external_data)
        
        # Generate visualizations
        generate_visualizations(risks, interaction_matrix, simulation_results, 
                                sensitivity_results, time_series_results)
        
        # Generate mitigation strategies
        mitigation_strategies = suggest_mitigation_strategies(risks, scenario_impacts, 
                                                              simulation_results)
        
        # Generate and save report
        report = generate_report(risks, categorized_risks, interaction_matrix, 
                                 scenario_impacts, simulation_results, clustered_risks, 
                                 risk_entities, sensitivity_results, time_series_results, 
                                 mitigation_strategies)

        logger.info("Risk Assessment Report generated successfully.")
        logger.info(f"Report saved to: {os.path.join(OUTPUT_DIR, 'climate_risk_report.json')}")
        logger.info(f"Visualizations saved in: {OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"An error occurred during the risk assessment process: {str(e)}")
        raise

if __name__ == "__main__":
    main()