from typing import Dict

CITY_COORDINATES: Dict[str, tuple] = {
    "lahore": (31.558, 74.351),
    "karachi": (24.8607, 67.0011),
    "islamabad": (33.6844, 73.0479),
    "delhi": (28.6139, 77.2090),
    "beijing": (39.9042, 116.4074),
    "new york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "los angeles": (34.0522, -118.2437),
}

def resolve_city(city: str) -> tuple:
    """Look up (lat, lon) for a known city name. Raises a clear error for an
    unknown one instead of silently failing deep inside an API call."""
    key = city.strip().lower()
    if key not in CITY_COORDINATES:
        known = ", ".join(sorted(CITY_COORDINATES))
        raise ValueError(f"Unknown city '{city}'. Known cities: {known}. "
                          f"Add it to CITY_COORDINATES to use it.")
    return CITY_COORDINATES[key]