"""Pipeline multi-agents : détection -> confiance -> orchestration -> recommandation.
Les résultats sont écrits dans le graphe (data/inferred.ttl) avec leur provenance."""
from agents import DetectorAgent, Orchestrator, RecommenderAgent, TrustAgent
from kg import ROOT, inferred_graph, load_graph, measurements_df


def main(strategy="trust_aware"):
    g = load_graph()
    df = measurements_df(g)
    print(f"[Graph]        {len(g)} triples, {len(df)} mesures lues via SPARQL")

    detector, trust_agent = DetectorAgent(), TrustAgent()
    df, events = detector.run(df)
    print(f"[{detector.name}]  {len(events)} événements détectés (Isolation Forest)")

    trust = trust_agent.run(df, events)
    print(f"[{trust_agent.name}]     confiance calculée : "
          f"{ {k: float(v) for k, v in trust.items()} }")

    orch = Orchestrator(strategy)
    events = RecommenderAgent().run(orch.run(events, trust))
    print(f"[{orch.name}]  stratégie = {strategy}\n")

    out = inferred_graph(events, trust)
    (ROOT / "data" / "inferred.ttl").write_text(out.serialize(format="turtle"), encoding="utf-8")

    for status in ("confirmed", "pending_verification"):
        sub = events[events["status"] == status]
        print(f"=== {status.upper()} ({len(sub)}) ===")
        for e in sub.itertuples():
            print(f"  {e.appliance:16} {e.start:%d/%m %Hh}-{e.end:%Hh}  "
                  f"x{e.severity:<5} conf={e.confidence}  -> {e.recommendation}")
        print()
    print(f"{len(out)} triples inférés écrits dans data/inferred.ttl")


if __name__ == "__main__":
    main()
