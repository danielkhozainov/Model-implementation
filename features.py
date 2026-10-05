"""
NorthCart — Batch Features Module
================================
Единая логика расчёта батч-признаков для исторических и инференсных срезов.

Все признаки считаются строго по данным до run_date (без утечки из будущего).
Окна: 7 дней (оперативное) и 30 дней (стабилизированное).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

SHORT_WINDOW = 7    # дней
LONG_WINDOW = 30    # дней

FEATURE_COLUMNS = [
    "customer_id",
    "run_date",
    # Просмотры страниц
    "page_view_7d",
    "page_view_30d",
    # Добавления в корзину
    "add_to_cart_7d",
    "add_to_cart_30d",
    # Конверсии
    "conv_view_to_cart_7d",
    "conv_view_to_cart_30d",
    "conv_cart_to_purchase_7d",
    "conv_cart_to_purchase_30d",
    # Уникальные товары
    "unique_products_7d",
    "unique_products_30d",
    # Сессии
    "sessions_7d",
    "sessions_30d",
    "avg_session_length_30d",
    # Заказы
    "days_since_last_purchase",
    "orders_30d",
    "sum_total_usd_30d",
    "avg_total_usd_30d",
]

# ---------------------------------------------------------------------------
# Загрузка исходных таблиц
# ---------------------------------------------------------------------------

def load_source_tables(
    engine: Engine,
    run_date: pd.Timestamp,
    window_days: int = LONG_WINDOW,
) -> dict[str, pd.DataFrame]:
    """
    Загружает из PostgreSQL все таблицы, нужные для расчёта признаков,
    ограничивая данные окном [run_date - window_days, run_date).

    Возвращает словарь DataFrame: customers, sessions, events, orders.
    """
    start_date = run_date - pd.Timedelta(days=window_days)

    # customers — все клиенты (не фильтруем по дате, нужны для базы)
    logger.info("Loading customers...")
    customers = pd.read_sql_query(
        text("SELECT customer_id, signup_date FROM customers"),
        engine,
    )

    # sessions — за окно
    logger.info("Loading sessions from %s to %s...", start_date.date(), run_date.date())
    sessions = pd.read_sql_query(
        text(
            "SELECT session_id, customer_id, start_time "
            "FROM sessions "
            "WHERE start_time >= :start_date AND start_time < :end_date"
        ),
        engine,
        params={"start_date": start_date, "end_date": run_date},
    )

    # events — через join к sessions, чтобы ограничить по времени
    logger.info("Loading events...")
    if sessions.empty:
        events = pd.DataFrame(
            columns=["event_id", "session_id", "timestamp", "event_type", "product_id"]
        )
    else:
        session_ids = sessions["session_id"].unique()
        # Загружаем пачками, чтобы не упереться в лимиты
        batch_size = 50_000
        chunks = []
        for i in range(0, len(session_ids), batch_size):
            batch = session_ids[i : i + batch_size]
            placeholders = ",".join([f":sid_{j}" for j in range(len(batch))])
            params = {f"sid_{j}": int(sid) for j, sid in enumerate(batch)}
            params["start_date"] = start_date
            params["end_date"] = run_date
            chunk = pd.read_sql_query(
                text(
                    f"SELECT event_id, session_id, timestamp, event_type, product_id "
                    f"FROM events "
                    f"WHERE session_id IN ({placeholders}) "
                    f"AND timestamp >= :start_date AND timestamp < :end_date"
                ),
                engine,
                params=params,
            )
            chunks.append(chunk)
        events = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
            columns=["event_id", "session_id", "timestamp", "event_type", "product_id"]
        )

    # orders — за окно + все заказы для days_since_last_purchase
    logger.info("Loading orders...")
    orders = pd.read_sql_query(
        text(
            "SELECT order_id, customer_id, order_time, total_usd "
            "FROM orders "
            "WHERE order_time < :end_date"
        ),
        engine,
        params={"end_date": run_date},
    )

    logger.info(
        "Loaded: customers=%d, sessions=%d, events=%d, orders=%d",
        len(customers), len(sessions), len(events), len(orders),
    )

    return {
        "customers": customers,
        "sessions": sessions,
        "events": events,
        "orders": orders,
    }


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _safe_div(a: float | int, b: float | int) -> float:
    """Безопасное деление: 0 если знаменатель равен 0."""
    return float(a) / float(b) if b else 0.0


def _aggregate_events(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    prefix: str,
) -> pd.DataFrame:
    """
    Агрегирует события по customer_id для заданного окна.
    Возвращает DataFrame с колонками:
      customer_id, {prefix}_page_view, {prefix}_add_to_cart,
      {prefix}_unique_products, {prefix}_sessions, {prefix}_avg_session_length,
      {prefix}_conv_view_to_cart, {prefix}_conv_cart_to_purchase
    """
    if events.empty or sessions.empty:
        return pd.DataFrame(columns=[
            "customer_id",
            f"{prefix}_page_view",
            f"{prefix}_add_to_cart",
            f"{prefix}_unique_products",
            f"{prefix}_sessions",
            f"{prefix}_avg_session_length",
            f"{prefix}_conv_view_to_cart",
            f"{prefix}_conv_cart_to_purchase",
        ])

    # Фильтруем события по окну
    ev = events[
        (events["timestamp"] >= window_start) & (events["timestamp"] < window_end)
    ].copy()

    # Фильтруем сессии по окну
    sess = sessions[
        (sessions["start_time"] >= window_start) & (sessions["start_time"] < window_end)
    ].copy()

    if ev.empty and sess.empty:
        return pd.DataFrame(columns=[
            "customer_id",
            f"{prefix}_page_view",
            f"{prefix}_add_to_cart",
            f"{prefix}_unique_products",
            f"{prefix}_sessions",
            f"{prefix}_avg_session_length",
            f"{prefix}_conv_view_to_cart",
            f"{prefix}_conv_cart_to_purchase",
        ])

    # Присоединяем customer_id к событиям через sessions
    sess_map = sess[["session_id", "customer_id", "start_time"]].copy()
    ev = ev.merge(sess_map[["session_id", "customer_id"]], on="session_id", how="left")
    ev = ev.dropna(subset=["customer_id"])

    # Длительность сессии (грубо: max timestamp - min timestamp в сессии)
    if not ev.empty:
        sess_duration = (
            ev.groupby("session_id")["timestamp"]
            .agg(["min", "max"])
            .reset_index()
        )
        sess_duration["duration"] = (
            sess_duration["max"] - sess_duration["min"]
        ).dt.total_seconds()
        sess_duration = sess_duration.merge(
            sess_map[["session_id", "customer_id"]], on="session_id", how="left"
        )
    else:
        sess_duration = pd.DataFrame(
            columns=["session_id", "min", "max", "duration", "customer_id"]
        )

    # Агрегации по customer_id
    result = pd.DataFrame()

    # Просмотры страниц
    page_views = ev[ev["event_type"] == "page_view"].groupby("customer_id").size()
    result[f"{prefix}_page_view"] = page_views

    # Добавления в корзину
    add_to_cart = ev[ev["event_type"] == "add_to_cart"].groupby("customer_id").size()
    result[f"{prefix}_add_to_cart"] = add_to_cart

    # Уникальные товары
    unique_products = (
        ev[ev["product_id"].notna()]
        .groupby("customer_id")["product_id"]
        .nunique()
    )
    result[f"{prefix}_unique_products"] = unique_products

    # Количество сессий
    sessions_count = sess.groupby("customer_id").size()
    result[f"{prefix}_sessions"] = sessions_count

    # Средняя длительность сессии
    if not sess_duration.empty:
        avg_sess_len = sess_duration.groupby("customer_id")["duration"].mean()
        result[f"{prefix}_avg_session_length"] = avg_sess_len
    else:
        result[f"{prefix}_avg_session_length"] = pd.Series(dtype=float)

    # Конверсия view → cart
    result[f"{prefix}_conv_view_to_cart"] = result.apply(
        lambda row: _safe_div(
            row.get(f"{prefix}_add_to_cart", 0),
            row.get(f"{prefix}_page_view", 0),
        ),
        axis=1,
    )

    # Конверсия cart → purchase (потребуются orders)
    # Пока ставим 0, заполним позже
    result[f"{prefix}_conv_cart_to_purchase"] = 0.0

    result = result.reset_index()
    return result


def _aggregate_orders(
    orders: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    run_date: pd.Timestamp,
    add_to_cart_by_customer: pd.Series | None,
    prefix: str,
) -> pd.DataFrame:
    """
    Агрегирует заказы по customer_id для заданного окна.
    Возвращает DataFrame с колонками:
      customer_id, {prefix}_orders, {prefix}_sum_total_usd,
      {prefix}_avg_total_usd, {prefix}_conv_cart_to_purchase,
      days_since_last_purchase
    """
    if orders.empty:
        return pd.DataFrame(columns=[
            "customer_id",
            f"{prefix}_orders",
            f"{prefix}_sum_total_usd",
            f"{prefix}_avg_total_usd",
            f"{prefix}_conv_cart_to_purchase",
            "days_since_last_purchase",
        ])

    # Заказы в окне
    orders_window = orders[
        (orders["order_time"] >= window_start) & (orders["order_time"] < window_end)
    ].copy()

    result = pd.DataFrame()

    if not orders_window.empty:
        orders_agg = orders_window.groupby("customer_id").agg(
            **{
                f"{prefix}_orders": ("order_id", "count"),
                f"{prefix}_sum_total_usd": ("total_usd", "sum"),
                f"{prefix}_avg_total_usd": ("total_usd", "mean"),
            }
        )
        result = orders_agg

    # Конверсия cart → purchase
    if add_to_cart_by_customer is not None and not orders_window.empty:
        cart_counts = add_to_cart_by_customer
        order_counts = orders_window.groupby("customer_id").size()
        conv = cart_counts.apply(lambda c: _safe_div(
            order_counts.get(cart_counts.index[cart_counts == c].name, 0), c
        ))
        # Проще: по каждому customer_id
        conv_series = pd.Series(dtype=float)
        for cid in cart_counts.index:
            conv_series[cid] = _safe_div(
                order_counts.get(cid, 0),
                cart_counts.get(cid, 0),
            )
        result[f"{prefix}_conv_cart_to_purchase"] = conv_series
    else:
        result[f"{prefix}_conv_cart_to_purchase"] = 0.0

    # days_since_last_purchase — по всем заказам (не только окно)
    all_orders = orders[orders["order_time"] < run_date].copy()
    if not all_orders.empty:
        last_purchase = all_orders.groupby("customer_id")["order_time"].max()
        days_since = (run_date - last_purchase).dt.days
        result["days_since_last_purchase"] = days_since

    result = result.reset_index()
    return result


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def build_batch_features(
    run_date: pd.Timestamp,
    engine: Engine,
    customer_ids: list[int] | None = None,
) -> pd.DataFrame:
    """
    Главная функция: рассчитывает батч-признаки для всех (или указанных) клиентов
    на дату среза run_date.

    Параметры:
        run_date: дата среза (T). Признаки считаются по данным строго до T.
        engine: SQLAlchemy engine для подключения к PostgreSQL.
        customer_ids: если указан — фильтрует результат по этим customer_id
                      (используется для инференса).

    Возвращает:
        DataFrame с колонками customer_id, run_date и 17 признаков.
    """
    logger.info("Building batch features for run_date=%s", run_date.date())

    # 1. Загружаем данные
    raw = load_source_tables(engine, run_date, window_days=LONG_WINDOW)
    customers = raw["customers"]
    sessions = raw["sessions"]
    events = raw["events"]
    orders = raw["orders"]

    # 2. Базовый список customer_id
    if customer_ids is not None:
        base_ids = set(customer_ids)
        logger.info("Filtering to %d customer_ids (inference mode)", len(base_ids))
    else:
        # Все клиенты, у которых есть хоть какая-то активность в окне
        active_in_sessions = set(sessions["customer_id"].unique()) if not sessions.empty else set()
        active_in_orders = set(orders["customer_id"].unique()) if not orders.empty else set()
        base_ids = active_in_sessions | active_in_orders
        logger.info("Active customers in window: %d", len(base_ids))

    if not base_ids:
        logger.warning("No active customers found for run_date=%s", run_date.date())
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    base_df = pd.DataFrame({"customer_id": sorted(base_ids)})

    # 3. Окна
    short_start = run_date - pd.Timedelta(days=SHORT_WINDOW)
    long_start = run_date - pd.Timedelta(days=LONG_WINDOW)

    # 4. Агрегации для короткого окна (7d)
    short_events = _aggregate_events(events, sessions, short_start, run_date, "short")
    short_orders = _aggregate_orders(
        orders, short_start, run_date, run_date,
        short_events.set_index("customer_id")["short_add_to_cart"]
        if "short_add_to_cart" in short_events.columns and not short_events.empty
        else None,
        "short",
    )

    # 5. Агрегации для длинного окна (30d)
    long_events = _aggregate_events(events, sessions, long_start, run_date, "long")
    long_orders = _aggregate_orders(
        orders, long_start, run_date, run_date,
        long_events.set_index("customer_id")["long_add_to_cart"]
        if "long_add_to_cart" in long_events.columns and not long_events.empty
        else None,
        "long",
    )

    # 6. Сборка финального датафрейма
    result = base_df.copy()

    # Короткое окно
    if not short_events.empty:
        short_events = short_events.rename(columns={
            "short_page_view": "page_view_7d",
            "short_add_to_cart": "add_to_cart_7d",
            "short_unique_products": "unique_products_7d",
            "short_sessions": "sessions_7d",
            "short_conv_view_to_cart": "conv_view_to_cart_7d",
            "short_conv_cart_to_purchase": "conv_cart_to_purchase_7d",
        })
        result = result.merge(short_events, on="customer_id", how="left")

    if not short_orders.empty:
        short_orders_renamed = short_orders.rename(columns={
            "short_orders": "orders_7d",
            "short_sum_total_usd": "sum_total_usd_7d",
            "short_avg_total_usd": "avg_total_usd_7d",
            "short_conv_cart_to_purchase": "conv_cart_to_purchase_7d_orders",
        })
        result = result.merge(short_orders_renamed, on="customer_id", how="left")

    # Длинное окно
    if not long_events.empty:
        long_events = long_events.rename(columns={
            "long_page_view": "page_view_30d",
            "long_add_to_cart": "add_to_cart_30d",
            "long_unique_products": "unique_products_30d",
            "long_sessions": "sessions_30d",
            "long_avg_session_length": "avg_session_length_30d",
            "long_conv_view_to_cart": "conv_view_to_cart_30d",
            "long_conv_cart_to_purchase": "conv_cart_to_purchase_30d",
        })
        result = result.merge(long_events, on="customer_id", how="left")

    if not long_orders.empty:
        long_orders_renamed = long_orders.rename(columns={
            "long_orders": "orders_30d",
            "long_sum_total_usd": "sum_total_usd_30d",
            "long_avg_total_usd": "avg_total_usd_30d",
            "long_conv_cart_to_purchase": "conv_cart_to_purchase_30d_orders",
        })
        result = result.merge(long_orders_renamed, on="customer_id", how="left")

    # 7. Заполняем пропуски
    numeric_fill = {
        "page_view_7d": 0, "page_view_30d": 0,
        "add_to_cart_7d": 0, "add_to_cart_30d": 0,
        "conv_view_to_cart_7d": 0.0, "conv_view_to_cart_30d": 0.0,
        "conv_cart_to_purchase_7d": 0.0, "conv_cart_to_purchase_30d": 0.0,
        "unique_products_7d": 0, "unique_products_30d": 0,
        "sessions_7d": 0, "sessions_30d": 0,
        "avg_session_length_30d": 0.0,
        "days_since_last_purchase": 0,
        "orders_30d": 0,
        "sum_total_usd_30d": 0.0,
        "avg_total_usd_30d": 0.0,
    }
    for col, fill_val in numeric_fill.items():
        if col in result.columns:
            result[col] = result[col].fillna(fill_val)

    # 8. Добавляем run_date
    result["run_date"] = run_date.strftime("%Y-%m-%d")

    # 9. Если есть лишние колонки — убираем
    keep = [c for c in FEATURE_COLUMNS if c in result.columns]
    result = result[keep]

    logger.info("Built features: %d rows, %d columns", len(result), len(result.columns))
    return result


# ---------------------------------------------------------------------------
# Валидация
# ---------------------------------------------------------------------------

def validate_features(
    features: pd.DataFrame,
    run_date: pd.Timestamp,
    min_rows: int = 1,
) -> None:
    """
    Проверяет, что датафрейм признаков корректен.
    Поднимает ValueError, если что-то не так.
    """
    if features.empty:
        raise ValueError("Features DataFrame is empty!")

    if len(features) < min_rows:
        raise ValueError(
            f"Too few rows: {len(features)} (expected >= {min_rows})"
        )

    # Проверяем обязательные колонки
    required = {"customer_id", "run_date"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Проверяем, что run_date в данных совпадает с ожидаемой
    if features["run_date"].nunique() != 1:
        raise ValueError(
            f"Multiple run_date values found: {features['run_date'].unique()}"
        )

    expected_date_str = run_date.strftime("%Y-%m-%d")
    actual_date = str(features["run_date"].iloc[0])
    if actual_date != expected_date_str:
        raise ValueError(
            f"run_date mismatch: expected {expected_date_str}, got {actual_date}"
        )

    # Проверяем отсутствие NaN в ключевых признаках
    key_features = [
        "page_view_7d", "page_view_30d",
        "add_to_cart_7d", "add_to_cart_30d",
        "sessions_7d", "sessions_30d",
    ]
    for col in key_features:
        if col in features.columns and features[col].isna().any():
            raise ValueError(f"NaN values found in column: {col}")

    # Проверяем отсутствие дубликатов customer_id
    if features["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer_id found!")

    logger.info("Validation passed: %d rows, %d columns", len(features), len(features.columns))


# ---------------------------------------------------------------------------
# Загрузка customer_id из исторических срезов (для инференса)
# ---------------------------------------------------------------------------

def load_historical_customer_ids(
    s3_client: Any,
    bucket: str,
    prefix: str = "batch_features/historical",
) -> list[int]:
    """
    Загружает уникальные customer_id из всех исторических срезов в S3.
    Используется для инференса, чтобы отфильтровать результат по тем же клиентам.
    """
    import io
    import json

    paginator = s3_client.get_paginator("list_objects_v2")
    customer_ids = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".csv") and not key.endswith(".parquet"):
                continue

            response = s3_client.get_object(Bucket=bucket, Key=key)
            body = response["Body"].read()

            if key.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(body))
            else:
                df = pd.read_parquet(io.BytesIO(body))

            if "customer_id" in df.columns:
                customer_ids.update(df["customer_id"].unique())

    logger.info("Loaded %d unique customer_ids from historical slices", len(customer_ids))
    return sorted(customer_ids)
