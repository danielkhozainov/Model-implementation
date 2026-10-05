# Batch Features Pipeline

Пайплайн расчёта батч-признаков для платформы интернет-торговли NorthCart.
Реализован как DAG в Apache Airflow. Признаки рассчитываются по единой логике
для исторических срезов (обучение модели) и инференсного среза (эксплуатация).

## Структура репозитория

```
airflow-project-template/
├── dags/
│   ├── batch_features.py              
│   └── calculate_batch_features.py
├── plugins/
│   └── features.py     
├── requirements.txt                     
├── README.md                           
├── Загрузка данных.ipynb              
└── Тетрадь проверки.ipynb             
```

**Назначение ключевых файлов:**
- `dags/batch_features.py` — DAG с задачами `build_and_upload_features` и `validate_saved_result`. Берёт `run_date` из Airflow Variable `batch_features_run_date`, считает признаки через модуль и загружает результат в S3.
- `dags/calculate_batch_features.py` — ядро логики. Содержит функции `build_batch_features`, `validate_features`, загрузку исходных таблиц, расчёт окон 7/30 дней, агрегацию по `customer_id`. Не содержит секретов — все подключения берутся из Airflow Connections.
- `Загрузка данных.ipynb` — тетрадь для локального осмотра исходных таблиц и отладки SQL-запросов.
- `Тетрадь проверки.ipynb` — локальная проверка результатов в S3, сравнение двух срезов, проверка пересечения `customer_id` и различий в признаках.

---

## Логика работы

### Параметр `run_date`

Ключевой параметр — дата среза T. Все признаки рассчитываются **строго по данным,
доступным до T**, что исключает утечку данных из будущего.

- **Оперативное окно** — 7 дней до T: `[T-7, T)`
- **Стабилизированное окно** — 30 дней до T: `[T-30, T)`

### Временные окна

Для каждого среза признаки считаются в двух окнах:
- **7 дней** (оперативное) — `[T-7, T)` — краткосрочное поведение
- **30 дней** (стабилизированное) — `[T-30, T)` — устойчивые паттерны

### Типы срезов

| Тип | Назначение | Окна | Ограничение |
|-----|-----------|------|-------------|
| Исторический | Обучение модели | 7d, 30d | После T должно быть ≥7 дней данных для целевой переменной |
| Инференсный | Скоринг, проверка качества | 7d, 30d | На тех же customer_id, что в исторических |

### Список признаков

| Признак | Описание | Окно |
|---------|----------|------|
| `page_view_7d` / `page_view_30d` | Количество просмотров страниц | 7d, 30d |
| `add_to_cart_7d` / `add_to_cart_30d` | Количество добавлений в корзину | 7d, 30d |
| `conv_view_to_cart_7d` / `conv_view_to_cart_30d` | Конверсия view → cart | 7d, 30d |
| `conv_cart_to_purchase_7d` / `conv_cart_to_purchase_30d` | Конверсия cart → purchase | 7d, 30d |
| `unique_products_7d` / `unique_products_30d` | Уникальные товары | 7d, 30d |
| `avg_session_length_30d` | Средняя длина сессии, сек | 30d |
| `sessions_7d` / `sessions_30d` | Количество сессий | 7d, 30d |
| `days_since_last_purchase` | Дней с последней покупки | — |
| `orders_30d` | Количество заказов | 30d |
| `sum_total_usd_30d` | Сумма заказов | 30d |
| `avg_total_usd_30d` | Средний чек | 30d |

## Схема данных

```
customers (customer_id, signup_date, ...)
     │
     ├──< sessions (session_id, customer_id, start_time, ...)
     │        │
     │        └──< events (event_id, session_id, timestamp, event_type, product_id)
     │
     └──< orders (order_id, customer_id, order_time, total_usd)
```

---

## Установка и настройка

### 1. Зависимости

```bash
pip install -r requirements.txt
```

### 2. Настройка Airflow

#### Переменные (Variables)

Создайте в Airflow (Admin → Variables):

| Ключ | Значение | Описание |
|------|----------|----------|
| `batch_features_run_date` | `2025-09-01` | Дата среза для расчёта признаков. Меняйте перед каждым запуском. |

#### Соединения (Connections)

Создайте соединения в Airflow (Admin → Connections):

**PostgreSQL:**

| Параметр | Значение |
|----------|----------|
| Conn Id | `student_db_connection` |
| Conn Type | `Postgres` |
| Host | `rc1d-pdq3kahro9enkfir.mdb.yandexcloud.net` |
| Schema | `playground_ds_20260915_7e2a105f57` |
| Login | `login` |
| Password | `password` |
| Port | `6432` |

> **Имя базы данных:** `playground_ds_20260915_7e2a105f57`  
> **Пользователь БД:** `ds_20260915_7e2a105f57`

Эти данные нужны ревьюеру для проверки решения.

**S3 (Yandex Object Storage):**

| Параметр | Значение |
|----------|----------|
| Conn Id | `student_s3` |
| Conn Type | `Amazon Web Services` |
| AWS Access Key ID | `Ключ доступа (хранится в Connection)` |
| AWS Secret Access Key | `Password (хранится в Connection)` |
| Extra | `{"bucket": "s3-ds-20260915-7e2a105f57", "endpoint_url": "https://storage.yandexcloud.net"}` |

### 3. Размещение файлов

Файлы `dags/batch_features.py` и `dags/calculate_batch_features.py` должны находиться
в папке `dags/` Airflow. Модуль `calculate_batch_features.py` импортируется
непосредственно из `batch_features.py` через `from calculate_batch_features import ...`.

---

## Запуск

### Через UI Airflow

1. Откройте Airflow UI → DAGs → `batch_features`.
2. Убедитесь, что в Variable `batch_features_run_date` установлена нужная дата.
3. Нажмите **Trigger DAG**.
4. Дождитесь, пока обе задачи станут зелёными.

### Исторические срезы (обучение)

Установите `batch_features_run_date = 2025-09-01` и запустите DAG.
Затем измените значение на `2025-10-01` и запустите снова.

### Инференсный срез

Установите `batch_features_run_date` на нужную дату инференса и запустите DAG.
Логика расчёта не меняется — меняется только дата среза.

---

## Куда сохраняется результат

Результат сохраняется в Yandex Object Storage (S3):

- **Бакет:** `s3-ds-20260915-7e2a105f57`
- **Путь:** `run_date=<YYYY-MM-DD>/batch_features.csv`

Примеры:
- `s3://s3-ds-20260915-7e2a105f57/run_date=2025-09-01/batch_features.csv`
- `s3://s3-ds-20260915-7e2a105f57/run_date=2025-10-01/batch_features.csv`

Каждый файл — CSV-таблица с колонками `customer_id`, `run_date` и набором признаков.
Результаты разных прогонов лежат раздельно и не перезаписывают друг друга.

---

## DAG: задачи

```
build_and_upload_features → validate_saved_result
```

| Задача | Описание |
|--------|----------|
| `build_and_upload_features` | Чтение `run_date` из Variable, загрузка данных из PostgreSQL, расчёт признаков через модуль `calculate_batch_features`, валидация, сохранение в S3 |
| `validate_saved_result` | Проверка наличия файла в S3, что файл не пустой и количество строк > 0 |

---

## Как проверить, что DAG отработал корректно

### Способ 1: через CLI

```bash
aws --endpoint-url=https://storage.yandexcloud.net s3 ls s3://s3-ds-20260915-7e2a105f57/
```

Убедитесь, что появились папки `run_date=2025-09-01/` и `run_date=2025-10-01/`,
в каждой — файл `batch_features.csv`.

### Способ 2: через тетрадь `Тетрадь проверки.ipynb`

Ожидаемые результаты:
- Набор колонок одинаковый в обоих файлах.
- Значения признаков различаются (так как окна разные).

---

## Принципы проектирования

1. **Единая логика** — модуль `calculate_batch_features.py` используется одинаково для обучения и инференса. Меняется только `run_date`.
2. **Отсутствие утечки данных** — все признаки считаются по данным строго до `run_date`.
3. **Воспроизводимость** — один `run_date` всегда даёт одинаковый результат.
4. **Разделение** — признаки и модель в разных циклах: признаки пересчитываются по расписанию без переобучения.
5. **Секреты** — пароли и ключи хранятся в Airflow Connections, не в коде.
6. **Изоляция прогонов** — каждый результат сохраняется по отдельному пути `run_date=<дата>/`.

---

## Технические параметры DAG

| Параметр | Значение | Назначение |
|----------|----------|------------|
| `schedule` | `None` | Ручной запуск |
| `catchup` | `False` | Без бэктилла |
| `start_date` | `2025-01-01` | Дата начала |
| `max_active_runs` | `1` | Один активный прогон за раз |

---

