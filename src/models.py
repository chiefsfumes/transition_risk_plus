from dataclasses import dataclass

@dataclass
class Risk:
    id: int
    description: str
    category: str
    likelihood: float
    impact: float

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "likelihood": self.likelihood,
            "impact": self.impact
        }

@dataclass
class ExternalData:
    year: int
    gdp_growth: float
    population: int
    energy_demand: float