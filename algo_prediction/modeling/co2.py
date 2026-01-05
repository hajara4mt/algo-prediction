# modeling/co2.py

CO2_COEF = {
    "elec": 0.0,  # TODO: mettre les vrais coeffs
    "gaz": 0.0,   # TODO: mettre les vrais coeffs
}

def get_co2_coef(fluid: str) -> float:
    return float(CO2_COEF.get(str(fluid).lower(), 0.0))
