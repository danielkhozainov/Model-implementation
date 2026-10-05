import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import Dict, Optional

#Загружаем исходные таблицы 
def load_source_tables(engine) -> Dict[str, pd.DataFrame]:

    tables = ["customers", "sessions", "events", "orders"]
    dfs = {}
    for t in tables:
        query = f"SELECT * FROM {t}"
        dfs[t] = pd.read_sql_query(query, engine)
    return dfs

#предобработка
def preprocess_data(raw: Dict[str, pd.DataFrame], run_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    
    cust = raw["customers"].copy()
    sess = raw["sessions"].copy()
    evt = raw["events"].copy()
    ord_ = raw["orders"].copy()

    #Приведение дат
    cust["signup_date"] = pd.to_datetime(cust["signup_date"], errors="coerce")
    sess["start_time"] = pd.to_datetime(sess["start_time"], errors="coerce")
    evt["timestamp"] = pd.to_datetime(evt["timestamp"], errors="coerce")
    ord_["order_time"] = pd.to_datetime(ord_["order_time"], errors="coerce")

    #Удаление строк с невалидными датами
    cust = cust[cust["signup_date"].notna()]
    sess = sess[sess["start_time"].notna()]
    evt = evt[evt["timestamp"].notna()]
    ord_ = ord_[ord_["order_time"].notna()]

    #Дубликаты по ключам
    sess = sess.drop_duplicates(subset=["session_id"])
    evt = evt.drop_duplicates(subset=["event_id"])
    ord_ = ord_.drop_duplicates(subset=["order_id"])

    #Фильтрация по run_date 
    cust = cust[cust["signup_date"] < run_date]
    sess = sess[sess["start_time"] < run_date]
    evt = evt[evt["timestamp"] < run_date]
    ord_ = ord_[ord_["order_time"] < run_date]

    #Мерджим
    evt = evt.merge(sess[["session_id", "customer_id"]], on="session_id", how="left")

    #events только с product_id для товарных признаков
    evt_with_product = evt[evt["product_id"].notna()].copy()

    #Денежные признаки приводим к нужным типам
    ord_["total_usd"] = pd.to_numeric(ord_["total_usd"], errors="coerce").fillna(0.0)

    return {
        "customers": cust,
        "sessions": sess,
        "events": evt,
        "events_with_product": evt_with_product,
        "orders": ord_,
    }

#Возвращает строки в окне run_date - days, run_date
def _filter_window(df: pd.DataFrame, ts_col: str, run_date: pd.Timestamp, days: int) -> pd.DataFrame:
    
    start = run_date - pd.Timedelta(days=days)
    return df[(df[ts_col] >= start) & (df[ts_col] < run_date)]

#Считает признаки по событиям (page_view, add_to_cart, purchase, уникальные товары)
def add_event_window_features(
    processed: Dict[str, pd.DataFrame],
    run_date: pd.Timestamp,
) -> pd.DataFrame:
    
    evt = processed["events"]
    evt_prod = processed["events_with_product"]

    evt_7 = _filter_window(evt, "timestamp", run_date, 7)
    evt_30 = _filter_window(evt, "timestamp", run_date, 30)
    evt_prod_7 = _filter_window(evt_prod, "timestamp", run_date, 7)
    evt_prod_30 = _filter_window(evt_prod, "timestamp", run_date, 30)

    def count_event(df, event_type):
        return df[df["event_type"] == event_type].groupby("customer_id").size()

    page_view_7 = count_event(evt_7, "page_view")
    page_view_30 = count_event(evt_30, "page_view")
    add_to_cart_7 = count_event(evt_7, "add_to_cart")
    add_to_cart_30 = count_event(evt_30, "add_to_cart")
    purchase_7 = count_event(evt_7, "purchase")
    purchase_30 = count_event(evt_30, "purchase")

    #Конверсии с обработкой деления на ноль
    conv_pv_atc_7 = (add_to_cart_7 / page_view_7).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    conv_pv_atc_30 = (add_to_cart_30 / page_view_30).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    conv_atc_pu_7 = (purchase_7 / add_to_cart_7).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    conv_atc_pu_30 = (purchase_30 / add_to_cart_30).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    unique_products_7 = evt_prod_7.groupby("customer_id")["product_id"].nunique()
    unique_products_30 = evt_prod_30.groupby("customer_id")["product_id"].nunique()

    features = pd.concat(
        [
            page_view_7.rename("page_view_count_7d"),
            page_view_30.rename("page_view_count_30d"),
            add_to_cart_7.rename("add_to_cart_count_7d"),
            add_to_cart_30.rename("add_to_cart_count_30d"),
            conv_pv_atc_7.rename("conversion_pv_to_atc_7d"),
            conv_pv_atc_30.rename("conversion_pv_to_atc_30d"),
            conv_atc_pu_7.rename("conversion_atc_to_purchase_7d"),
            conv_atc_pu_30.rename("conversion_atc_to_purchase_30d"),
            unique_products_7.rename("unique_product_id_count_7d"),
            unique_products_30.rename("unique_product_id_count_30d"),
        ],
        axis=1,
    ).fillna(0)

    return features

#Признаки по сессиям (длина, количество)
def add_session_features(
    processed: Dict[str, pd.DataFrame],
    run_date: pd.Timestamp,
) -> pd.DataFrame:
    
    sess = processed["sessions"]

    #Расчёт длины сессии по событиям
    evt = processed["events"]
    session_len = (
        evt.groupby("session_id")["timestamp"]
        .agg(["min", "max"])
        .assign(session_length_sec=lambda x: (x["max"] - x["min"]).dt.total_seconds())[["session_length_sec"]]
        .reset_index()
    )

    sess_full = sess.merge(session_len, on="session_id", how="left").fillna({"session_length_sec": 0})

    sess_7 = _filter_window(sess_full, "start_time", run_date, 7)
    sess_30 = _filter_window(sess_full, "start_time", run_date, 30)

    sessions_count_7 = sess_7.groupby("customer_id").size()
    sessions_count_30 = sess_30.groupby("customer_id").size()
    avg_session_length_30 = sess_30.groupby("customer_id")["session_length_sec"].mean()

    features = pd.concat(
        [
            avg_session_length_30.rename("avg_session_length_30d"),
            sessions_count_7.rename("sessions_count_7d"),
            sessions_count_30.rename("sessions_count_30d"),
        ],
        axis=1,
    ).fillna(0)

    return features

#Признаки по заказам
def add_order_features(
    processed: Dict[str, pd.DataFrame],
    run_date: pd.Timestamp,
) -> pd.DataFrame:
    
    ord_ = processed["orders"]

    ord_30 = _filter_window(ord_, "order_time", run_date, 30)

    # days_since_last_purchase: по всем заказам до run_date
    last_purchase = ord_.groupby("customer_id")["order_time"].max()
    days_since_last = (run_date - last_purchase).dt.days

    all_cust_ids = processed["customers"]["customer_id"].unique()
    days_since_last = days_since_last.reindex(all_cust_ids, fill_value=-1)

    order_count_30 = ord_30.groupby("customer_id").size()
    total_usd_sum_30 = ord_30.groupby("customer_id")["total_usd"].sum()
    total_usd_avg_30 = ord_30.groupby("customer_id")["total_usd"].mean()

    features = pd.concat(
        [
            days_since_last.rename("days_since_last_purchase"),
            order_count_30.rename("order_count_30d"),
            total_usd_sum_30.rename("total_usd_sum_30d"),
            total_usd_avg_30.rename("total_usd_avg_30d"),
        ],
        axis=1,
    ).fillna(0)

    return features

#Итоговая таблица
def build_batch_features(run_date: pd.Timestamp, engine, customer_ids: Optional[list] = None) -> pd.DataFrame:
    
    raw = load_source_tables(engine)
    processed = preprocess_data(raw, run_date)

    feat_events = add_event_window_features(processed, run_date)
    feat_sessions = add_session_features(processed, run_date)
    feat_orders = add_order_features(processed, run_date)

    features = (
        feat_events.join(feat_sessions, how="outer")
        .join(feat_orders, how="outer")
        .fillna(0)
        .reset_index()
        .rename(columns={"index": "customer_id"})
    )

    features["run_date"] = run_date.date()

    if customer_ids is not None:
        features = features[features["customer_id"].isin(customer_ids)]

    #Одна строка на customer_id
    assert features.duplicated(subset=["customer_id"]).sum() == 0, "Дубликаты customer_id!"

    return features

#Сохраняем таблицу
def save_features(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)


def validate_features(df: pd.DataFrame, run_date: pd.Timestamp) -> None:
    
    #Нет пропусков
    na_count = df.isna().sum().sum()
    assert na_count == 0, f"Есть пропуски: {na_count}"

    #Конверсий не больше 1
    conv_cols = [c for c in df.columns if c.startswith("conversion_")]
    for c in conv_cols:
        assert (df[c] >= 0).all() and (df[c] <= 1).all(), f"Конверсия {c} вне диапазона [0,1]"

    #Деление на ноль 
    pv_zero = df[df["page_view_count_7d"] == 0]
    assert (pv_zero["conversion_pv_to_atc_7d"] == 0).all(), "Конверсия pv->atc не 0 при page_view=0"

    atc_zero = df[df["add_to_cart_count_7d"] == 0]
    assert (atc_zero["conversion_atc_to_purchase_7d"] == 0).all(), "Конверсия atc->purchase не 0 при add_to_cart=0"

    #days_since_last_purchase присваиваем -1 только если нет заказов вообще
    no_orders = df[df["days_since_last_purchase"] == -1]
    assert (no_orders["order_count_30d"] == 0).all(), "У пользователей с -1 есть заказы за 30 дней"

