"""Agents : détection (IA), confiance, orchestration, recommandation."""
import pandas as pd
from sklearn.ensemble import IsolationForest

TRANSIENT_H = 2      # un événement <= 2 h est jugé "transitoire" (défaut capteur probable)
TRUST_THRESHOLD = 0.8
MIN_DEVIATION = 0.35   # écart minimal à la normale pour retenir une mesure atypique


class DetectorAgent:
    """Détecte les mesures atypiques (Isolation Forest + seuil d'écart), puis les regroupe en événements."""

    name = "DetectorAgent"

    def __init__(self, contamination="auto", seed=42):
        self.model = IsolationForest(n_estimators=200, contamination=contamination,
                                     random_state=seed)

    def run(self, df):
        df = df.copy()
        # normalisation par appareil : comparable entre appareils de puissances différentes
        df["ratio"] = df["value"] / df.groupby("appliance")["value"].transform("median")
        atypical = self.model.fit_predict(df[["ratio"]].values) == -1
        # l'IA propose, un seuil d'écart minimal écarte les variations bénignes
        df["flag"] = atypical & ((df["ratio"] - 1).abs() > MIN_DEVIATION)
        return df, self._events(df)

    @staticmethod
    def _events(df):
        events = []
        for (appliance, sensor), grp in df.groupby(["appliance", "sensor"]):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            run_id = (grp["flag"] != grp["flag"].shift()).cumsum()
            for _, run in grp.groupby(run_id):
                if not run["flag"].iloc[0]:
                    continue
                events.append({
                    "appliance": appliance, "sensor": sensor,
                    "start": run["timestamp"].iloc[0], "end": run["timestamp"].iloc[-1],
                    "duration_h": len(run), "severity": round(run["ratio"].mean(), 2),
                })
        ev = pd.DataFrame(events)
        if ev.empty:
            return pd.DataFrame(columns=["event_id", "appliance", "sensor", "start", "end",
                                         "duration_h", "severity"])
        ev = ev.sort_values(["appliance", "start"]).reset_index(drop=True)
        ev.insert(0, "event_id", [f"Event_{r.appliance}_{i:03d}" for i, r in ev.iterrows()])
        return ev


class TrustAgent:
    """Calcule un score de confiance par capteur.

    Hypothèse : les événements courts et isolés (pics, pertes de signal) révèlent un défaut
    du capteur ; une déviation soutenue reflète plutôt un vrai phénomène.
    trust = 1 - 2 * (heures en événements transitoires / nombre de mesures)
    """

    name = "TrustAgent"

    def run(self, df, events):
        trust = {}
        for sensor, grp in df.groupby("sensor"):
            ev = events[(events["sensor"] == sensor) & (events["duration_h"] <= TRANSIENT_H)]
            rate = ev["duration_h"].sum() / len(grp)
            trust[sensor] = round(max(0.0, 1 - 2 * rate), 2)
        return trust


class Orchestrator:
    """Décide, pour chaque événement, d'agir ou de vérifier d'abord."""

    name = "Orchestrator"

    def __init__(self, strategy="trust_aware", threshold=TRUST_THRESHOLD):
        assert strategy in ("naive", "trust_aware")
        self.strategy, self.threshold = strategy, threshold

    def run(self, events, trust):
        ev = events.copy()
        ev["confidence"] = ev["sensor"].map(trust)
        if self.strategy == "naive":
            ev["status"] = "confirmed"
        else:
            ev["status"] = ev["confidence"].apply(
                lambda t: "confirmed" if t >= self.threshold else "pending_verification")
        return ev


class RecommenderAgent:
    """Associe une action à chaque événement, selon son statut et sa nature."""

    name = "RecommenderAgent"

    def run(self, events):
        ev = events.copy()
        ev["recommendation"] = ev.apply(self._advice, axis=1) if len(ev) else []
        return ev

    @staticmethod
    def _advice(e):
        if e["status"] == "pending_verification":
            return f"Verify sensor {e['sensor']} (low trust) before any action on {e['appliance']}."
        if e["severity"] < 0.2:
            return f"Signal loss on {e['appliance']}: check sensor wiring."
        if e["duration_h"] > TRANSIENT_H:
            return f"Sustained over-consumption on {e['appliance']}: schedule an inspection."
        return f"Short spike on {e['appliance']}: monitor."
