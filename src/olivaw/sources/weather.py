from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from olivaw.sources.base import SourceHealth, SourcePayload

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_LABELS = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class WeatherProvider(Protocol):
    def fetch_forecast(
        self,
        *,
        latitude: float,
        longitude: float,
        units: str,
    ) -> dict[str, object]:
        """Fetch provider-native weather data."""


@dataclass(frozen=True)
class OpenMeteoProvider:
    timeout: float = 5.0

    def fetch_forecast(
        self,
        *,
        latitude: float,
        longitude: float,
        units: str,
    ) -> dict[str, object]:
        unit = _temperature_unit(units)
        wind_unit = "mph" if unit == "fahrenheit" else "kmh"
        query = urllib.parse.urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "daily": ",".join(
                    (
                        "weather_code",
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                    )
                ),
                "temperature_unit": unit,
                "wind_speed_unit": wind_unit,
                "timezone": "auto",
                "forecast_days": 1,
            }
        )
        request = urllib.request.Request(
            f"{OPEN_METEO_URL}?{query}",
            headers={"User-Agent": "olivaw-weather-source/1.0"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read()
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Open-Meteo response was not an object")
        return data


@dataclass(frozen=True)
class WeatherSource:
    enabled: bool = False
    latitude: float | None = None
    longitude: float | None = None
    location_name: str | None = None
    units: str = "fahrenheit"
    provider: WeatherProvider | None = None
    source_id: str = "weather"
    display_name: str = "Weather"

    def health(self) -> SourceHealth:
        if not self.enabled:
            return SourceHealth(
                source_id=self.source_id,
                display_name=self.display_name,
                status="unavailable",
                message="Weather source is disabled.",
            )
        if self.latitude is None or self.longitude is None:
            return SourceHealth(
                source_id=self.source_id,
                display_name=self.display_name,
                status="unavailable",
                message="Weather source requires latitude and longitude.",
            )
        return SourceHealth(
            source_id=self.source_id,
            display_name=self.display_name,
            status="ok",
            message=f"Weather configured for {self._location_label()}.",
        )

    def fetch(self) -> SourcePayload:
        health = self.health()
        if health.status != "ok":
            return {
                "source": self.source_id,
                "status": health.status,
                "count": 0,
                "items": [],
                "diagnostics": self._diagnostics(
                    provider_status="not called",
                    last_fetch_status=health.status,
                ),
            }

        try:
            raw = (self.provider or OpenMeteoProvider()).fetch_forecast(
                latitude=self.latitude or 0.0,
                longitude=self.longitude or 0.0,
                units=self.units,
            )
        except Exception as exc:
            return {
                "source": self.source_id,
                "status": "error",
                "count": 0,
                "items": [],
                "errors": [f"Weather fetch failed: {type(exc).__name__}: {exc}"],
                "diagnostics": self._diagnostics(
                    provider_status="error",
                    last_fetch_status=f"{type(exc).__name__}: {exc}",
                ),
            }

        item = self._normalize(raw)
        return {
            "source": self.source_id,
            "status": "ok",
            "count": 1,
            "items": [item],
            "diagnostics": self._diagnostics(
                provider_status="ok",
                last_fetch_status="ok",
                forecast_date=item.get("forecast_date"),
            ),
        }

    def _normalize(self, raw: dict[str, object]) -> dict[str, object]:
        current = _dict(raw.get("current"))
        current_units = _dict(raw.get("current_units"))
        daily = _dict(raw.get("daily"))
        daily_units = _dict(raw.get("daily_units"))
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        forecast_date = _first_list_text(daily.get("time"))
        current_temperature = _number(current.get("temperature_2m"))
        high_temperature = _first_list_number(daily.get("temperature_2m_max"))
        low_temperature = _first_list_number(daily.get("temperature_2m_min"))
        rain_chance = _first_list_number(daily.get("precipitation_probability_max"))
        wind_speed = _number(current.get("wind_speed_10m"))
        condition_code = _first_int(
            current.get("weather_code"),
            _first_list_value(daily.get("weather_code")),
        )
        condition = WEATHER_CODE_LABELS.get(
            condition_code if condition_code is not None else -1,
            "unknown",
        )
        temperature_unit = str(
            current_units.get("temperature_2m")
            or daily_units.get("temperature_2m_max")
            or _temperature_symbol(self.units)
        )
        wind_unit = str(current_units.get("wind_speed_10m") or "")
        summary = _weather_summary(
            current_temperature=current_temperature,
            high_temperature=high_temperature,
            low_temperature=low_temperature,
            rain_chance=rain_chance,
            condition=condition,
            temperature_unit=temperature_unit,
        )
        facts = [
            _fact("current_temperature", f"Current temperature: {_format_degree(current_temperature, temperature_unit)}"),
            _fact("condition", f"Condition: {condition}"),
            _fact("high_temperature", f"High temperature: {_format_degree(high_temperature, temperature_unit)}"),
            _fact("low_temperature", f"Low temperature: {_format_degree(low_temperature, temperature_unit)}"),
            _fact("precipitation_chance", f"Rain chance: {_format_percent(rain_chance)}"),
        ]
        if wind_speed is not None:
            facts.append(_fact("wind", f"Wind: {_format_number(wind_speed)} {wind_unit}".strip()))
        return {
            "title": self._location_label(),
            "summary": summary,
            "generated_at": generated_at,
            "observed_at": str(current.get("time") or forecast_date or ""),
            "forecast_date": forecast_date,
            "forecast_window": forecast_date,
            "location_name": self.location_name or "",
            "latitude": self.latitude,
            "longitude": self.longitude,
            "provider": "Open-Meteo",
            "condition": condition,
            "temperature_unit": temperature_unit,
            "current_temperature": current_temperature,
            "high_temperature": high_temperature,
            "low_temperature": low_temperature,
            "precipitation_chance": rain_chance,
            "wind_speed": wind_speed,
            "wind_unit": wind_unit,
            "facts": [fact for fact in facts if fact["summary"]],
        }

    def _diagnostics(
        self,
        *,
        provider_status: str,
        last_fetch_status: str,
        forecast_date: object = "",
    ) -> dict[str, object]:
        return {
            "enabled": "yes" if self.enabled else "no",
            "configured": "yes" if self.latitude is not None and self.longitude is not None else "no",
            "provider": "Open-Meteo",
            "provider_status": provider_status,
            "location_name": self.location_name or "",
            "latitude": self.latitude,
            "longitude": self.longitude,
            "units": _temperature_unit(self.units),
            "last_fetch_status": last_fetch_status,
            "forecast_date": forecast_date or "",
        }

    def _location_label(self) -> str:
        return self.location_name or "Configured weather location"


def _weather_summary(
    *,
    current_temperature: float | None,
    high_temperature: float | None,
    low_temperature: float | None,
    rain_chance: float | None,
    condition: str,
    temperature_unit: str,
) -> str:
    parts = [
        f"Currently {_format_degree(current_temperature, temperature_unit)} and {condition}.",
        f"High {_format_degree(high_temperature, temperature_unit)}, low {_format_degree(low_temperature, temperature_unit)}.",
        f"Rain chance {_format_percent(rain_chance)}.",
    ]
    return " ".join(part for part in parts if "unknown" not in part.lower())


def _fact(kind: str, summary: str) -> dict[str, object]:
    if "unknown" in summary.lower():
        summary = ""
    return {"kind": kind, "summary": summary}


def _temperature_unit(units: str) -> str:
    normalized = units.strip().lower()
    return "celsius" if normalized in {"celsius", "metric", "c"} else "fahrenheit"


def _temperature_symbol(units: str) -> str:
    return "°C" if _temperature_unit(units) == "celsius" else "°F"


def _format_degree(value: float | None, unit: str) -> str:
    if value is None:
        return "unknown"
    return f"{_format_number(value)}{unit}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{_format_number(value)}%"


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _number(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _first_list_value(value: object) -> object:
    if isinstance(value, list) and value:
        return value[0]
    return None


def _first_list_text(value: object) -> str:
    item = _first_list_value(value)
    return str(item or "").strip()


def _first_list_number(value: object) -> float | None:
    return _number(_first_list_value(value))


def _first_int(*values: object) -> int | None:
    for value in values:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            continue
    return None
