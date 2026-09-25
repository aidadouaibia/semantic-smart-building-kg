"""Exécute toutes les requêtes SPARQL du dossier queries/ sur le graphe."""
from pathlib import Path
from rdflib import Graph

ROOT = Path(__file__).resolve().parent.parent

DATA_FILES = [
    "ontology/smart_building.ttl",
    "data/building.ttl",
    "data/measurements.ttl",  # généré par simulate.py
    "data/inferred.ttl",      # généré par run_pipeline.py
]


def load(graph, path):
    # Lecture manuelle : évite le bug Windows "unknown url type: e"
    graph.parse(data=path.read_text(encoding="utf-8"), format="turtle")


def short(term):
    return str(term).split("#")[-1] if term is not None else "-"


def main():
    g = Graph()
    for f in DATA_FILES:
        p = ROOT / f
        if p.exists():
            load(g, p)
        else:
            print(f"(fichier absent, ignoré : {f})")
    print(f"Graph loaded: {len(g)} triples\n")

    for qfile in sorted((ROOT / "queries").glob("*.rq")):
        print(f"=== {qfile.name} ===")
        try:
            for row in g.query(qfile.read_text(encoding="utf-8")):
                print("  " + " | ".join(short(v) for v in row))
        except Exception as e:
            print(f"  ERREUR dans {qfile.name} : {e}")
        print()


if __name__ == "__main__":
    main()
