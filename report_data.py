from datetime import datetime, timedelta
from pathlib import Path
import random
import sqlite3

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "instance" / "road_reports.sqlite3"
SEED = 20260929

AREAS = [
    ("Dindigul Town", "Dindigul", "Urban", 215000),
    ("Natham", "Dindigul", "Town", 55000),
    ("Vedasandur", "Dindigul", "Town", 48000),
    ("Palani", "Dindigul", "Town", 72000),
    ("Oddanchatram", "Dindigul", "Town", 52000),
    ("Nilakottai", "Dindigul", "Town", 42000),
    ("Batlagundu", "Dindigul", "Town", 38000),
    ("Athoor", "Dindigul", "Rural", 61000),
    ("Gujiliamparai", "Dindigul", "Rural", 33000),
    ("Kodaikanal", "Dindigul", "Hill town", 36000),
]

VEHICLE_TYPES = [
    ("Two-wheeler", 48),
    ("Car / van", 22),
    ("Bus", 8),
    ("Truck", 9),
    ("Auto-rickshaw", 9),
    ("Pedestrian / cyclist", 4),
]


def _connect():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_reports_database():
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS report_areas (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                district TEXT NOT NULL,
                settlement_type TEXT NOT NULL,
                population_estimate INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS report_incidents (
                id INTEGER PRIMARY KEY,
                area_id INTEGER NOT NULL REFERENCES report_areas(id),
                occurred_at TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                deaths INTEGER NOT NULL CHECK (deaths >= 0),
                source TEXT NOT NULL DEFAULT 'Synthetic demo record'
            );
            CREATE INDEX IF NOT EXISTS idx_report_incidents_time_area
                ON report_incidents(occurred_at, area_id);
            """
        )
        area_count = connection.execute("SELECT COUNT(*) FROM report_areas").fetchone()[0]
        if area_count == 0:
            connection.executemany(
                "INSERT INTO report_areas(name, district, settlement_type, population_estimate) VALUES (?, ?, ?, ?)",
                AREAS,
            )

        incident_count = connection.execute("SELECT COUNT(*) FROM report_incidents").fetchone()[0]
        if incident_count == 0:
            _seed_incidents(connection)


def _seed_incidents(connection):
    rng = random.Random(SEED)
    area_ids = [row[0] for row in connection.execute("SELECT id FROM report_areas ORDER BY id")]
    area_weights = [26, 9, 8, 13, 10, 7, 6, 8, 5, 8]
    vehicle_names = [vehicle[0] for vehicle in VEHICLE_TYPES]
    vehicle_weights = [vehicle[1] for vehicle in VEHICLE_TYPES]
    start = datetime(2024, 1, 1)
    end = datetime(2026, 9, 29, 23, 59)
    span_minutes = int((end - start).total_seconds() // 60)
    records = []

    for incident_id in range(1, 361):
        occurred_at = start + timedelta(minutes=rng.randrange(span_minutes + 1))
        area_id = rng.choices(area_ids, weights=area_weights, k=1)[0]
        vehicle_type = rng.choices(vehicle_names, weights=vehicle_weights, k=1)[0]
        death_roll = rng.random()
        if vehicle_type == "Pedestrian / cyclist":
            deaths = 2 if death_roll < 0.08 else 1 if death_roll < 0.37 else 0
        else:
            deaths = 2 if death_roll < 0.025 else 1 if death_roll < 0.16 else 0
        records.append(
            (
                incident_id,
                area_id,
                occurred_at.strftime("%Y-%m-%d %H:%M:%S"),
                vehicle_type,
                deaths,
                "Synthetic demo record",
            )
        )

    records.sort(key=lambda record: record[2])
    connection.executemany(
        "INSERT INTO report_incidents(id, area_id, occurred_at, vehicle_type, deaths, source) VALUES (?, ?, ?, ?, ?, ?)",
        records,
    )


def get_report_areas():
    with _connect() as connection:
        return [dict(row) for row in connection.execute("SELECT * FROM report_areas ORDER BY name")]


def get_area_period_report(start_datetime, end_exclusive, area_id=None):
    query = """
        SELECT a.id, a.name, a.district, a.settlement_type, a.population_estimate,
               COUNT(i.id) AS accident_count,
               COALESCE(SUM(i.deaths), 0) AS deaths
        FROM report_areas AS a
        LEFT JOIN report_incidents AS i
          ON i.area_id = a.id AND i.occurred_at >= ? AND i.occurred_at < ?
    """
    parameters = [start_datetime, end_exclusive]
    if area_id is not None:
        query += " WHERE a.id = ?"
        parameters.append(area_id)
    query += " GROUP BY a.id ORDER BY accident_count DESC, a.name"

    with _connect() as connection:
        rows = [dict(row) for row in connection.execute(query, parameters)]
    return rows


def get_incident_report(start_datetime, end_exclusive, area_id=None):
    query = """
        SELECT i.id, i.occurred_at, a.name AS area, a.settlement_type,
               i.vehicle_type, i.deaths, i.source
        FROM report_incidents AS i
        JOIN report_areas AS a ON a.id = i.area_id
        WHERE i.occurred_at >= ? AND i.occurred_at < ?
    """
    parameters = [start_datetime, end_exclusive]
    if area_id is not None:
        query += " AND a.id = ?"
        parameters.append(area_id)
    query += " ORDER BY i.occurred_at DESC, i.id DESC"

    with _connect() as connection:
        return [dict(row) for row in connection.execute(query, parameters)]
