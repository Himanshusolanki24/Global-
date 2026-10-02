"""₹ per call. Local: idle-subtracted kWh × tariff. API: provider usage tokens × list price."""

from __future__ import annotations

from bench.core.schemas import ApiPrice


def local_cost_inr(energy_wh: float | None, tariff_inr_per_kwh: float) -> float | None:
    # ponytail: electricity only; add GPU amortisation (₹/h × total_ms) if the bench moves off free Kaggle GPUs
    return None if energy_wh is None else energy_wh / 1000 * tariff_inr_per_kwh


def api_cost_inr(input_tokens: int, output_tokens: int, price: ApiPrice, usd_inr: float) -> float:
    usd = (input_tokens * price.in_usd_per_m + output_tokens * price.out_usd_per_m) / 1e6
    return usd * usd_inr
