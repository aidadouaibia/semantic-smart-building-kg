"""Génère data/measurements.ttl et data/ground_truth.json à partir de data/building.ttl."""
import json

from kg import ROOT, load_graph, measurements_graph
from simulation import simulate

building = load_graph(["data/building.ttl"])
records, truth = simulate(building)
graph = measurements_graph(records)

(ROOT / "data" / "measurements.ttl").write_text(graph.serialize(format="turtle"), encoding="utf-8")
(ROOT / "data" / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
print(f"{len(graph)} triples ecrits | {len(truth['sensor_faults'])} defauts capteur "
      f"| {len(truth['real_anomalies'])} vraies anomalies")
