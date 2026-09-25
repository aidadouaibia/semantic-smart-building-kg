"""Simulateur de mesures (en mémoire) : consommation normale + capteurs défaillants + vraies anomalies."""
import math
import random
from datetime import datetime, timedelta

from kg import local

START = datetime(2026, 9, 14, 0, 0)
HOURS = 48
# Vraies anomalies : appareil -> (heure de début, heure de fin, facteur de surconsommation)
REAL_ANOMALIES = {"Fridge01": (30, 35, 1.8), "CoffeeMachine01": (20, 23, 2.5)}


def appliances_info(building_graph):
    """Lit (appareil, capteur, kWh/mois) depuis le graphe par SPARQL."""
    rows = building_graph.query(
        """PREFIX sb: <http://smartbuilding.org#>
           SELECT ?a ?s ?kwh WHERE {
             ?a sb:monitoredBy ?s ; sb:energyConsumption ?kwh . }"""
    )
    return sorted(((local(a), local(s), float(k)) for a, s, k in rows), key=lambda r: r[0])


def simulate(building_graph, faulty=("Sensor02",), fault_rate=0.25, seed=42,
             hours=HOURS, real_anomalies=REAL_ANOMALIES):
    """Renvoie (records, truth). fault_rate = part des mesures corrompues (60 % pics, 40 % pertes)."""
    rng = random.Random(seed)
    faulty = set(faulty)
    spike_p, zero_p = fault_rate * 0.6, fault_rate * 0.4
    records = []
    truth = {"real_anomalies": [], "sensor_faults": []}

    for name, sensor, kwh in appliances_info(building_graph):
        baseline = kwh * 1000 / 720  # kWh/mois -> Wh/heure
        window = real_anomalies.get(name)
        if window:
            truth["real_anomalies"].append({
                "appliance": name,
                "start": (START + timedelta(hours=window[0])).isoformat(),
                "end": (START + timedelta(hours=window[1])).isoformat(),
            })
        for h in range(hours):
            ts = START + timedelta(hours=h)
            daily = 1 + 0.15 * math.sin(2 * math.pi * (ts.hour - 6) / 24)
            value = baseline * daily * rng.gauss(1, 0.05)
            in_real = bool(window and window[0] <= h <= window[1])
            if in_real:
                value *= window[2]
            r = rng.random()
            spike = rng.uniform(2.5, 4)  # toujours tirés : flux aléatoire stable
            if sensor in faulty and not in_real:
                if r < spike_p:
                    value *= spike
                    truth["sensor_faults"].append(f"M_{name}_{h:03d}")
                elif r < spike_p + zero_p:
                    value = 0.0
                    truth["sensor_faults"].append(f"M_{name}_{h:03d}")
            records.append({"id": f"M_{name}_{h:03d}", "appliance": name, "sensor": sensor,
                            "value": round(value, 1), "timestamp": ts})
    return records, truth
