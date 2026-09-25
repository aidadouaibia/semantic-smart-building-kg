"""Accès au graphe de connaissances : chargement et lecture des mesures."""
from pathlib import Path

import pandas as pd
from rdflib import RDF, XSD, Graph, Literal, Namespace

ROOT = Path(__file__).resolve().parent.parent
SB = Namespace("http://smartbuilding.org#")
PROV = Namespace("http://www.w3.org/ns/prov#")

BASE_FILES = ["ontology/smart_building.ttl", "data/building.ttl", "data/measurements.ttl"]


def local(term):
    return str(term).split("#")[-1]


def load_graph(files=BASE_FILES):
    g = Graph()
    for f in files:
        p = ROOT / f
        if p.exists():
            # lecture manuelle : évite le bug de chemin Windows
            g.parse(data=p.read_text(encoding="utf-8"), format="turtle")
    return g


def measurements_df(g):
    """Lit les mesures du graphe via SPARQL -> DataFrame trié par appareil et temps."""
    rows = g.query(
        """PREFIX sb: <http://smartbuilding.org#>
           SELECT ?a ?s ?v ?t WHERE {
             ?m a sb:Measurement ; sb:measures ?a ; sb:producedBy ?s ;
                sb:value ?v ; sb:timestamp ?t . }"""
    )
    df = pd.DataFrame(
        [(local(a), local(s), float(v), t.toPython()) for a, s, v, t in rows],
        columns=["appliance", "sensor", "value", "timestamp"],
    )
    return df.sort_values(["appliance", "timestamp"]).reset_index(drop=True)


def measurements_graph(records):
    """Transforme des mesures (liste de dicts) en triples RDF."""
    out = Graph()
    out.bind("sb", SB)
    out.bind("xsd", XSD)
    for r in records:
        m = SB[r["id"]]
        out.add((m, RDF.type, SB.Measurement))
        out.add((m, SB.measures, SB[r["appliance"]]))
        out.add((m, SB.producedBy, SB[r["sensor"]]))
        out.add((m, SB.value, Literal(r["value"], datatype=XSD.float)))
        out.add((m, SB.timestamp, Literal(r["timestamp"].isoformat(), datatype=XSD.dateTime)))
    return out


def inferred_graph(events, trust):
    """Transforme les conclusions des agents (confiance, événements) en triples RDF + provenance."""
    out = Graph()
    out.bind("sb", SB)
    out.bind("prov", PROV)
    out.bind("xsd", XSD)
    for name in ("DetectorAgent", "TrustAgent", "Orchestrator", "RecommenderAgent"):
        out.add((SB[name], RDF.type, PROV.SoftwareAgent))
    for sensor, score in trust.items():
        out.add((SB[sensor], SB.computedTrust, Literal(float(score), datatype=XSD.float)))
        out.add((SB[sensor], PROV.wasAttributedTo, SB.TrustAgent))
    for e in events.itertuples():
        ev = SB[e.event_id]
        out.add((ev, RDF.type, SB.AnomalyEvent))
        out.add((ev, SB.affects, SB[e.appliance]))
        out.add((ev, PROV.wasDerivedFrom, SB[e.sensor]))
        out.add((ev, PROV.wasAttributedTo, SB.DetectorAgent))
        out.add((ev, SB.startTime, Literal(e.start.isoformat(), datatype=XSD.dateTime)))
        out.add((ev, SB.endTime, Literal(e.end.isoformat(), datatype=XSD.dateTime)))
        out.add((ev, SB.durationHours, Literal(int(e.duration_h), datatype=XSD.integer)))
        out.add((ev, SB.severity, Literal(float(e.severity), datatype=XSD.float)))
        out.add((ev, SB.confidence, Literal(float(e.confidence), datatype=XSD.float)))
        out.add((ev, SB.status, Literal(e.status)))
        out.add((ev, SB.recommendationText, Literal(e.recommendation)))
    return out
