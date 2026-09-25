"""Expérience : orchestration naïve vs orchestration sensible à la confiance."""
import pandas as pd

from agents import TRUST_THRESHOLD, DetectorAgent, Orchestrator, TrustAgent
from evaluate import label_events, load_truth, parse_truth, summarize
from kg import ROOT, load_graph, measurements_df
from simulation import simulate

STRATEGIES = ("naive", "trust_aware")


def run_once(df, windows, threshold=TRUST_THRESHOLD):
    """Exécute détection + confiance + les deux stratégies sur un jeu de mesures."""
    df, events = DetectorAgent().run(df)
    trust = TrustAgent().run(df, events)
    results = {s: summarize(label_events(Orchestrator(s, threshold).run(events, trust), windows),
                            windows) for s in STRATEGIES}
    return results, trust


def run_many(building_graph, seeds, faulty, fault_rate, threshold=TRUST_THRESHOLD):
    """Répète l'expérience sur plusieurs graines aléatoires. Renvoie un tableau par graine."""
    rows = []
    for seed in seeds:
        records, truth = simulate(building_graph, faulty, fault_rate, seed)
        df = pd.DataFrame(records)
        results, _ = run_once(df, parse_truth(truth), threshold)
        for strategy, m in results.items():
            rows.append({"seed": seed, "strategy": strategy, **m})
    return pd.DataFrame(rows)


def main():
    df = measurements_df(load_graph())
    windows = load_truth()
    results, trust = run_once(df, windows)
    print("Confiance calculée :", {k: float(v) for k, v in trust.items()}, "\n")

    table = pd.DataFrame(results)
    print(table.to_string())
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "experiment.csv")
    md = ["| metric | naive | trust_aware |", "|---|---|---|"]
    md += [f"| {m} | {r['naive']} | {r['trust_aware']} |" for m, r in table.iterrows()]
    (out / "experiment.md").write_text("\n".join(md), encoding="utf-8")

    building = load_graph(["data/building.ttl"])
    many = run_many(building, range(20), {"Sensor02"}, 0.25)
    summary = many.groupby("strategy")[["alerts_sent", "false_alarms", "precision",
                                        "real_anomalies_alerted", "real_events_deferred"]]
    print("\nMoyenne sur 20 graines :")
    print(summary.mean().round(2).to_string())
    summary.mean().round(2).to_csv(out / "experiment_20seeds.csv")


if __name__ == "__main__":
    main()
