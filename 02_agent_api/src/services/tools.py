from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """Devuelve el pronóstico del clima actual para la ubicación indicada."""
    return f"El clima en {location} es soleado con una temperatura de 22°C."
