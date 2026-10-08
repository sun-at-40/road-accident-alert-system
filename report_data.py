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


SCHEMA_VERSION = 2
IMAGE_COUNT = 40

VEHICLE_MODELS = {
    "Two-wheeler": ["Honda Activa 6G", "TVS Apache RTR 160", "Hero Splendor Plus", "Bajaj Pulsar 150", "Royal Enfield Classic 350"],
    "Car / van": ["Maruti Suzuki Swift", "Hyundai i20", "Toyota Innova Crysta", "Tata Nexon", "Mahindra Bolero"],
    "Bus": ["Ashok Leyland Viking", "Tata Starbus", "Eicher Skyline"],
    "Truck": ["Tata 407", "Ashok Leyland Boss", "BharatBenz 1617", "Eicher Pro 3015"],
    "Auto-rickshaw": ["Bajaj RE", "Piaggio Ape City", "Mahindra Alfa"],
    "Pedestrian / cyclist": ["Bicycle", "Pedestrian"],
}

DESCRIPTIONS = [
    "Vehicle skidded on a wet road and hit the median.",
    "Rear-end collision at a busy junction during peak hours.",
    "Head-on collision while overtaking on a two-lane road.",
    "Vehicle lost control on a curve and overturned.",
    "Collision at an unsignalled intersection.",
    "Vehicle hit a roadside barrier at high speed.",
    "Side-swipe collision between two vehicles in dense traffic.",
    "Driver lost control on a ghat road bend and hit the guard rail.",
    "Vehicle struck a pedestrian while crossing the road.",
    "Low-visibility collision near a village road crossing at night.",
]

LOCATIONS = [
    "Main Road junction", "Bus stand roundabout", "Bypass road", "NH-83 stretch", "Market street",
    "Railway gate crossing", "Government hospital road", "Bridge approach", "School zone road", "Ghat road bend",
]



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
            """
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(report_incidents)")}
        if columns and "vehicle_number" not in columns:
            # Synthetic data only: rebuild the table with the extended schema.
            connection.execute("DROP TABLE report_incidents")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS report_incidents (
                id INTEGER PRIMARY KEY,
                area_id INTEGER NOT NULL REFERENCES report_areas(id),
                occurred_at TEXT NOT NULL,
                location TEXT NOT NULL,
                vehicle_number TEXT,
                vehicle_type TEXT NOT NULL,
                vehicle_model TEXT NOT NULL,
                description TEXT NOT NULL,
                image_file TEXT NOT NULL,
                deaths INTEGER NOT NULL CHECK (deaths >= 0),
                source TEXT NOT NULL DEFAULT 'Synthetic demo record'
            );
            CREATE INDEX IF NOT EXISTS idx_report_incidents_time_area
                ON report_incidents(occurred_at, area_id);
            CREATE INDEX IF NOT EXISTS idx_report_incidents_vehicle
                ON report_incidents(vehicle_number);
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


def normalize_vehicle_number(value):
    return "".join(ch for ch in (value or "").upper() if ch.isalnum())


def _make_vehicle_numbers(rng, count):
    numbers = set()
    while len(numbers) < count:
        numbers.add(
            f"TN57{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{rng.randrange(1000, 10000)}"
        )
    return sorted(numbers)


def _seed_incidents(connection):
    rng = random.Random(SEED)
    areas = {row["id"]: row["name"] for row in connection.execute("SELECT id, name FROM report_areas ORDER BY id")}
    area_ids = list(areas)
    area_weights = [26, 9, 8, 13, 10, 7, 6, 8, 5, 8]
    vehicle_names = [vehicle[0] for vehicle in VEHICLE_TYPES]
    vehicle_weights = [vehicle[1] for vehicle in VEHICLE_TYPES]
    vehicle_pool = _make_vehicle_numbers(rng, 300)
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
            vehicle_number = None
            description = "Vehicle struck a pedestrian or cyclist on the road."
        else:
            deaths = 2 if death_roll < 0.025 else 1 if death_roll < 0.16 else 0
            vehicle_number = rng.choice(vehicle_pool)
            description = rng.choice(DESCRIPTIONS[:7] + DESCRIPTIONS[7:8] + DESCRIPTIONS[9:])
        location = f"{rng.choice(LOCATIONS)}, {areas[area_id]}"
        records.append(
            (
                incident_id,
                area_id,
                occurred_at.strftime("%Y-%m-%d %H:%M:%S"),
                location,
                vehicle_number,
                vehicle_type,
                rng.choice(VEHICLE_MODELS[vehicle_type]),
                description,
                f"accident-{rng.randrange(1, IMAGE_COUNT + 1):02d}.jpg",
                deaths,
                "Synthetic demo record",
            )
        )

    records.sort(key=lambda record: record[2])
    connection.executemany(
        """INSERT INTO report_incidents(id, area_id, occurred_at, location, vehicle_number, vehicle_type,
           vehicle_model, description, image_file, deaths, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        records,
    )


def search_vehicle_incidents(vehicle_number):
    normalized = normalize_vehicle_number(vehicle_number)
    if not normalized:
        return []
    query = """
        SELECT i.id, i.occurred_at, i.location, a.name AS area, i.vehicle_number, i.vehicle_type,
               i.vehicle_model, i.description, i.image_file, i.deaths, i.source
        FROM report_incidents AS i
        JOIN report_areas AS a ON a.id = i.area_id
        WHERE i.vehicle_number = ?
        ORDER BY i.occurred_at DESC
    """
    with _connect() as connection:
        return [dict(row) for row in connection.execute(query, (normalized,))]


def get_sample_vehicle_numbers(limit=4):
    with _connect() as connection:
        rows = connection.execute(
            """SELECT vehicle_number FROM report_incidents WHERE vehicle_number IS NOT NULL
               GROUP BY vehicle_number ORDER BY COUNT(*) DESC, vehicle_number LIMIT ?""",
            (limit,),
        )
        return [row[0] for row in rows]

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
        SELECT i.id, i.occurred_at, i.location, a.name AS area, a.settlement_type,
               i.vehicle_number, i.vehicle_type, i.vehicle_model, i.description, i.image_file, i.deaths, i.source
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
