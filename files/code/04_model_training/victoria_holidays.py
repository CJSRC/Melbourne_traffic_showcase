from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HolidayRecord:
    date: str
    name: str
    restricted_trading: bool = False


VICTORIA_PUBLIC_HOLIDAYS = {
    2020: [
        HolidayRecord("2020-01-01", "new_years_day"),
        HolidayRecord("2020-01-27", "australia_day_observed"),
        HolidayRecord("2020-03-09", "labour_day"),
        HolidayRecord("2020-04-10", "good_friday", True),
        HolidayRecord("2020-04-11", "easter_saturday"),
        HolidayRecord("2020-04-12", "easter_sunday"),
        HolidayRecord("2020-04-13", "easter_monday"),
        HolidayRecord("2020-04-25", "anzac_day", True),
        HolidayRecord("2020-06-08", "queens_birthday"),
        HolidayRecord("2020-10-23", "afl_grand_final_friday"),
        HolidayRecord("2020-11-03", "melbourne_cup"),
        HolidayRecord("2020-12-25", "christmas_day", True),
        HolidayRecord("2020-12-26", "boxing_day"),
        HolidayRecord("2020-12-28", "boxing_day_observed"),
    ],
    2022: [
        HolidayRecord("2022-01-01", "new_years_day"),
        HolidayRecord("2022-01-03", "new_years_day_observed"),
        HolidayRecord("2022-01-26", "australia_day"),
        HolidayRecord("2022-03-14", "labour_day"),
        HolidayRecord("2022-04-15", "good_friday", True),
        HolidayRecord("2022-04-16", "easter_saturday"),
        HolidayRecord("2022-04-17", "easter_sunday"),
        HolidayRecord("2022-04-18", "easter_monday"),
        HolidayRecord("2022-04-25", "anzac_day", True),
        HolidayRecord("2022-06-13", "queens_birthday"),
        HolidayRecord("2022-09-22", "national_day_of_mourning"),
        HolidayRecord("2022-09-23", "afl_grand_final_friday"),
        HolidayRecord("2022-11-01", "melbourne_cup"),
        HolidayRecord("2022-12-25", "christmas_day", True),
        HolidayRecord("2022-12-26", "boxing_day"),
        HolidayRecord("2022-12-27", "christmas_day_observed"),
    ],
    2023: [
        HolidayRecord("2023-01-01", "new_years_day"),
        HolidayRecord("2023-01-02", "new_years_day_observed"),
        HolidayRecord("2023-01-26", "australia_day"),
        HolidayRecord("2023-03-13", "labour_day"),
        HolidayRecord("2023-04-07", "good_friday", True),
        HolidayRecord("2023-04-08", "easter_saturday"),
        HolidayRecord("2023-04-09", "easter_sunday"),
        HolidayRecord("2023-04-10", "easter_monday"),
        HolidayRecord("2023-04-25", "anzac_day", True),
        HolidayRecord("2023-06-12", "kings_birthday"),
        HolidayRecord("2023-09-29", "afl_grand_final_friday"),
        HolidayRecord("2023-11-07", "melbourne_cup"),
        HolidayRecord("2023-12-25", "christmas_day", True),
        HolidayRecord("2023-12-26", "boxing_day"),
    ],
    2025: [
        HolidayRecord("2025-01-01", "new_years_day"),
        HolidayRecord("2025-01-27", "australia_day_observed"),
        HolidayRecord("2025-03-10", "labour_day"),
        HolidayRecord("2025-04-18", "good_friday", True),
        HolidayRecord("2025-04-19", "easter_saturday"),
        HolidayRecord("2025-04-20", "easter_sunday"),
        HolidayRecord("2025-04-21", "easter_monday"),
        HolidayRecord("2025-04-25", "anzac_day", True),
        HolidayRecord("2025-06-09", "kings_birthday"),
        HolidayRecord("2025-09-26", "afl_grand_final_friday"),
        HolidayRecord("2025-11-04", "melbourne_cup"),
        HolidayRecord("2025-12-25", "christmas_day", True),
        HolidayRecord("2025-12-26", "boxing_day"),
    ],
    2026: [
        HolidayRecord("2026-01-01", "new_years_day"),
        HolidayRecord("2026-01-26", "australia_day"),
        HolidayRecord("2026-03-09", "labour_day"),
        HolidayRecord("2026-04-03", "good_friday", True),
        HolidayRecord("2026-04-04", "easter_saturday"),
        HolidayRecord("2026-04-05", "easter_sunday"),
        HolidayRecord("2026-04-06", "easter_monday"),
        HolidayRecord("2026-04-25", "anzac_day", True),
        HolidayRecord("2026-06-08", "kings_birthday"),
        HolidayRecord("2026-11-03", "melbourne_cup"),
        HolidayRecord("2026-12-25", "christmas_day", True),
        HolidayRecord("2026-12-26", "boxing_day"),
        HolidayRecord("2026-12-28", "boxing_day_observed"),
    ],
}


def build_victoria_holiday_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, records in VICTORIA_PUBLIC_HOLIDAYS.items():
        for record in records:
            rows.append(
                {
                    "date": record.date,
                    "holiday_name": record.name,
                    "is_public_holiday": 1.0,
                    "is_restricted_trading_day": float(record.restricted_trading),
                    "year": int(year),
                }
            )
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out


def add_victoria_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    holidays = build_victoria_holiday_frame()
    out = out.merge(holidays, on="date", how="left")

    out["is_public_holiday"] = out["is_public_holiday"].fillna(0.0).astype("float32")
    out["is_restricted_trading_day"] = out["is_restricted_trading_day"].fillna(0.0).astype("float32")
    out["holiday_name"] = out["holiday_name"].fillna("none")

    dt = pd.to_datetime(out["date"])
    holiday_dates = set(holidays["date"].tolist())
    prev_dates = (dt - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    next_dates = (dt + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    out["is_holiday_eve"] = prev_dates.isin(holiday_dates).astype("float32")
    out["is_day_before_holiday"] = next_dates.isin(holiday_dates).astype("float32")
    out["is_melbourne_cup"] = (out["holiday_name"] == "melbourne_cup").astype("float32")
    out["is_afl_grand_final_friday"] = (out["holiday_name"] == "afl_grand_final_friday").astype("float32")
    out["is_easter_period"] = out["holiday_name"].str.contains("easter|good_friday", regex=True).astype("float32")
    out["is_christmas_period"] = out["holiday_name"].str.contains("christmas|boxing", regex=True).astype("float32")
    out["is_new_year_period"] = out["holiday_name"].str.contains("new_year", regex=True).astype("float32")
    out["is_australia_day"] = out["holiday_name"].str.contains("australia_day", regex=True).astype("float32")
    out["is_labour_day"] = (out["holiday_name"] == "labour_day").astype("float32")
    out["is_anzac_day"] = out["holiday_name"].str.contains("anzac", regex=True).astype("float32")
    out["is_monarch_birthday"] = out["holiday_name"].str.contains("birthday", regex=True).astype("float32")
    out["is_oneoff_national_event"] = (out["holiday_name"] == "national_day_of_mourning").astype("float32")
    return out
