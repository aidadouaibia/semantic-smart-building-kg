"""Compare les événements détectés à la vérité terrain."""
import json

import pandas as pd

from kg import ROOT


def parse_truth(truth):
    """Convertit la vérité terrain (dict) en fenêtres avec vrais Timestamps."""
    return [{"appliance": w["appliance"], "start": pd.Timestamp(w["start"]),
             "end": pd.Timestamp(w["end"])} for w in truth["real_anomalies"]]


def load_truth():
    path = ROOT / "data" / "ground_truth.json"
    return parse_truth(json.loads(path.read_text(encoding="utf-8")))


def _overlaps(r, w):
    return w["appliance"] == r.appliance and r.start <= w["end"] and r.end >= w["start"]


def label_events(events, windows):
    """Ajoute is_real : True si l'événement recoupe une vraie anomalie du même appareil."""
    ev = events.copy()
    ev["is_real"] = [any(_overlaps(r, w) for w in windows) for r in ev.itertuples()]
    return ev


def summarize(ev, windows):
    """Métriques d'une stratégie d'orchestration (ev doit contenir is_real)."""
    conf = ev[ev["status"] == "confirmed"]
    pend = ev[ev["status"] == "pending_verification"]
    real_ok = conf[conf["is_real"]]
    found = sum(any(_overlaps(r, w) for r in real_ok.itertuples()) for w in windows)
    return {
        "events_detected": len(ev),
        "alerts_sent": len(conf),
        "false_alarms": len(conf) - len(real_ok),
        "precision": round(len(real_ok) / len(conf), 2) if len(conf) else float("nan"),
        "real_anomalies_alerted": found,
        "real_anomalies_total": len(windows),
        "deferred_for_verification": len(pend),
        "real_events_deferred": int(pend["is_real"].sum()),
    }
