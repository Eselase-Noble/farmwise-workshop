"""Fetch current weather for the crop form, via Open-Meteo (free, no API key).

The crop model takes temperature (deg C), humidity (%) and rainfall (mm). This
module turns a place name (or coordinates) into those three numbers so a farmer
can auto-fill most of the form instead of measuring the weather by hand:

* temperature and humidity come from the current-conditions forecast;
* rainfall is the *average monthly* precipitation over the last ~12 months from
  the archive. The crop model was trained on rainfall values of roughly
  20-300 mm, so a real annual total (often 1000 mm+ in Ghana) would always fall
  outside its range and be flagged; the monthly average lands inside it and is
  the sensible figure to compare. It is best-effort: if the archive call fails
  the other two values are still returned.

All requests go to fixed Open-Meteo hosts with timeouts, so this never reaches
the local network and cannot hang the web app.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_TIMEOUT = (5, 15)  # (connect, read) seconds
# The archive is only needed for the best-effort rainfall figure and its first
# hit for a location can be slow, so it gets a tighter budget: better to leave
# rainfall blank for manual entry than to make the whole auto-fill crawl.
_ARCHIVE_TIMEOUT = (5, 10)


class WeatherError(RuntimeError):
    """A place could not be found or the weather service could not be reached."""


def _get(url: str, params: dict, timeout=_TIMEOUT) -> dict:
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise WeatherError("Could not reach the weather service. Check the connection.") from exc
    if not resp.ok:
        raise WeatherError(f"Weather service returned {resp.status_code}.")
    return resp.json()


def geocode(name: str) -> dict:
    """Resolve a place name to coordinates and a readable label."""
    data = _get(GEOCODE_URL, {"name": name, "count": 1, "language": "en", "format": "json"})
    results = data.get("results") or []
    if not results:
        raise WeatherError(f"Could not find a place called '{name}'.")
    top = results[0]
    label_parts = [top.get("name"), top.get("admin1"), top.get("country")]
    return {
        "latitude": top["latitude"],
        "longitude": top["longitude"],
        "label": ", ".join(p for p in label_parts if p),
    }


def _monthly_rainfall_mm(latitude: float, longitude: float) -> Optional[float]:
    """Average monthly precipitation over the last ~12 months, or None.

    The archive lags a few days, so the window ends a week back. We return the
    monthly mean (annual total / 12) because that matches the scale the crop
    model was trained on; see the module docstring.
    """
    end = dt.date.today() - dt.timedelta(days=7)
    start = end - dt.timedelta(days=365)
    try:
        data = _get(
            ARCHIVE_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": "precipitation_sum",
                "timezone": "auto",
            },
            timeout=_ARCHIVE_TIMEOUT,
        )
    except WeatherError:
        return None
    values = (data.get("daily") or {}).get("precipitation_sum") or []
    totals = [v for v in values if isinstance(v, (int, float))]
    return round(sum(totals) / 12, 1) if totals else None


def weather_for(
    *,
    location: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> dict:
    """Return temperature, humidity and rainfall for a place name or coordinates."""
    label = location or ""
    if latitude is None or longitude is None:
        if not location:
            raise WeatherError("Provide a place name or latitude and longitude.")
        geo = geocode(location)
        latitude, longitude, label = geo["latitude"], geo["longitude"], geo["label"]

    data = _get(
        FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m",
            "timezone": "auto",
        },
    )
    current = data.get("current") or {}
    temperature = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    if temperature is None or humidity is None:
        raise WeatherError("The weather service did not return current conditions.")

    return {
        "location": label,
        "latitude": latitude,
        "longitude": longitude,
        "temperature": round(float(temperature), 1),
        "humidity": round(float(humidity), 1),
        "rainfall": _monthly_rainfall_mm(latitude, longitude),
    }
