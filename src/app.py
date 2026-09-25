"""Interface Streamlit : Trust-aware multi-agent orchestration over a smart-building knowledge graph.
Lancer :  streamlit run src/app.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt
import pandas as pd
import streamlit as st

from agents import DetectorAgent, Orchestrator, RecommenderAgent, TrustAgent
from evaluate import label_events, parse_truth, summarize
from experiment import STRATEGIES, run_many
from kg import ROOT, inferred_graph, load_graph, local, measurements_graph
from llm_agent import PROVIDERS, LLMAgent
from simulation import appliances_info, simulate

st.set_page_config(page_title="Trust-aware Smart Building KG", page_icon="🏢", layout="wide")


# ---------------------------------------------------------------- données
@st.cache_resource
def base_graph():
    return load_graph(["ontology/smart_building.ttl", "data/building.ttl"])


def sensors_and_prior(g):
    rows = g.query("""PREFIX sb: <http://smartbuilding.org#>
                      SELECT ?s ?t WHERE { ?s a sb:Sensor . OPTIONAL { ?s sb:trustScore ?t } }""")
    return {local(s): (float(t) if t is not None else None) for s, t in rows}


@st.cache_data(show_spinner=False)
def run_scenario(faulty, fault_rate, seed, threshold):
    """Simulation + pipeline complet + évaluation pour les deux stratégies."""
    records, truth = simulate(base_graph(), faulty, fault_rate, seed)
    df = pd.DataFrame(records)
    windows = parse_truth(truth)
    df, events = DetectorAgent().run(df)
    trust = TrustAgent().run(df, events)
    by_strategy, summaries = {}, {}
    for s in STRATEGIES:
        ev = RecommenderAgent().run(Orchestrator(s, threshold).run(events, trust))
        by_strategy[s] = label_events(ev, windows)
        summaries[s] = summarize(by_strategy[s], windows)
    return df, {k: float(v) for k, v in trust.items()}, by_strategy, summaries, windows


@st.cache_data(show_spinner=False)
def robustness(faulty, fault_rate, threshold, n):
    return run_many(base_graph(), range(n), set(faulty), fault_rate, threshold)


# ---------------------------------------------------------------- barre latérale
g0 = base_graph()
prior = sensors_and_prior(g0)
info = appliances_info(g0)

st.sidebar.title("⚙️ Scénario")
faulty = st.sidebar.multiselect("Capteurs défaillants (injection de fautes)", sorted(prior),
                                default=["Sensor02"] if "Sensor02" in prior else [])
fault_rate = st.sidebar.slider("Taux de fautes des capteurs défaillants", 0.0, 0.5, 0.25, 0.05)
threshold = st.sidebar.slider("Seuil de confiance (orchestrateur)", 0.3, 0.95, 0.8, 0.05)
seed = st.sidebar.number_input("Graine aléatoire", 0, 9999, 42)
st.sidebar.caption("Changez un paramètre : les agents se relancent automatiquement.")

st.sidebar.divider()
st.sidebar.subheader("💬 Assistant")
_options = ["offline", "groq", "openrouter", "gemini", "mistral", "anthropic"]
_env_key = {p: os.getenv(c["env"]) for p, c in PROVIDERS.items()}
_env_key["anthropic"] = os.getenv("ANTHROPIC_API_KEY")
_default = next((p for p in _options[1:] if _env_key.get(p)), "offline")  # clé trouvée dans .env
provider = st.sidebar.selectbox(
    "Fournisseur", _options, index=_options.index(_default),
    format_func=lambda p: {"offline": "Hors ligne (sans clé)", "groq": "Groq (gratuit)",
                           "openrouter": "OpenRouter (gratuit)", "gemini": "Google Gemini (gratuit)",
                           "mistral": "Mistral (gratuit)", "anthropic": "Anthropic (payant)"}[p])
api_key, model = "", ""
if provider != "offline":
    api_key = st.sidebar.text_input("Clé API", type="password", key=f"key_{provider}",
                                    help="Laissez vide pour utiliser la clé du fichier .env.")
    if not api_key and _env_key.get(provider):
        st.sidebar.caption("🔑 Clé lue depuis le fichier .env")
    default_model = PROVIDERS[provider]["model"] if provider in PROVIDERS else "claude-sonnet-5"
    model = st.sidebar.text_input("Modèle", default_model, key=f"model_{provider}",
                                  help="Les noms de modèles gratuits changent : corrigez-le si besoin.")
agent = LLMAgent(api_key=api_key or None, model=model or None, provider=provider)
st.sidebar.caption(f"Mode actif : **{agent.mode}**" + (f" ({agent.label})" if agent.mode == "llm" else ""))

df, trust, ev_by, summ, windows = run_scenario(tuple(sorted(faulty)), fault_rate, int(seed), threshold)
ev_t = ev_by["trust_aware"]

st.title("🏢 Trust-aware multi-agent orchestration on a smart-building knowledge graph")
st.caption("Des agents lisent un graphe RDF, détectent des anomalies (Isolation Forest), "
           "évaluent la confiance des capteurs et décident d'agir ou de vérifier.")

c = st.columns(5)
c[0].metric("Appareils", len(info))
c[1].metric("Mesures", len(df))
c[2].metric("Événements détectés", len(ev_t))
c[3].metric("Alertes confirmées", int((ev_t["status"] == "confirmed").sum()))
c[4].metric("En vérification", int((ev_t["status"] == "pending_verification").sum()))

# graphe courant : ontologie + bâtiment + mesures simulées + conclusions des agents
records, _ = simulate(base_graph(), faulty, fault_rate, int(seed))
kg = load_graph(["ontology/smart_building.ttl", "data/building.ttl"])
kg += measurements_graph(records)
kg += inferred_graph(ev_t.drop(columns=["is_real"]), trust)

tab_dash, tab_trust, tab_ev, tab_exp, tab_chat, tab_sparql, tab_about = st.tabs(
    ["📈 Dashboard", "🛡️ Confiance & graphe", "🤖 Décisions des agents",
     "🧪 Expérience", "💬 Assistant", "🔎 Console SPARQL", "ℹ️ Architecture"])

# ---------------------------------------------------------------- dashboard
with tab_dash:
    show_truth = st.checkbox("Afficher la vérité terrain (zones vertes = vraies anomalies)")
    for name, sensor, _ in info:
        d = df[df["appliance"] == name]
        line = alt.Chart(d).mark_line(color="#4c78a8").encode(
            x=alt.X("timestamp:T", title=None), y=alt.Y("value:Q", title="Wh/h"))
        pts = alt.Chart(d[d["flag"]]).mark_circle(color="#e45756", size=80).encode(
            x="timestamp:T", y="value:Q", tooltip=["timestamp", "value", "ratio"])
        layers = [line, pts]
        if show_truth:
            w = pd.DataFrame([{"a": x["start"], "b": x["end"]} for x in windows
                              if x["appliance"] == name])
            if len(w):
                layers.insert(0, alt.Chart(w).mark_rect(color="#54a24b", opacity=0.2).encode(
                    x="a:T", x2="b:T"))
        t = trust.get(sensor, 1.0)
        st.markdown(f"**{name}** — capteur `{sensor}` (confiance calculée : {t:.2f}) — "
                    f"points rouges = mesures atypiques")
        st.altair_chart(alt.layer(*layers).properties(height=150), width="stretch")

# ---------------------------------------------------------------- confiance + graphe
with tab_trust:
    left, right = st.columns([1, 1.3])
    with left:
        rows = [{"Capteur": s, "Confiance a priori (manuelle)": prior[s],
                 "Confiance calculée (TrustAgent)": trust.get(s),
                 "Décision": "✅ fiable" if trust.get(s, 1) >= threshold else "⚠️ à vérifier"}
                for s in sorted(prior)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.info("Règle du TrustAgent : trust = 1 − 2 × (heures en événements courts et isolés / mesures). "
                "Un événement ≤ 2 h trahit plutôt un défaut du capteur ; une déviation soutenue "
                "reflète plutôt un vrai phénomène.")
    with right:
        colour = lambda t: "#54a24b" if t >= threshold else ("#f58518" if t >= 0.4 else "#e45756")
        dot = ["digraph G { rankdir=LR; node [fontname=Helvetica, style=filled, fontcolor=black];"]
        loc = {local(a): local(r) for a, r in g0.query(
            "PREFIX sb: <http://smartbuilding.org#> SELECT ?a ?r WHERE { ?a sb:locatedIn ?r }")}
        for r in sorted(set(loc.values())):
            dot.append(f'"{r}" [shape=box, fillcolor="#dddddd"];')
        for name, sensor, _ in info:
            nev = ev_t[ev_t["appliance"] == name]
            lab = f"{name}\\n{len(nev)} evt ({(nev['status'] == 'confirmed').sum()} conf.)"
            dot.append(f'"{name}" [shape=ellipse, fillcolor="#cfe2f3", label="{lab}"];')
            dot.append(f'"{name}" -> "{loc.get(name, "?")}" [label="locatedIn", fontsize=9];')
            dot.append(f'"{name}" -> "{sensor}" [label="monitoredBy", fontsize=9];')
        for s in sorted(prior):
            t = trust.get(s, 1.0)
            dot.append(f'"{s}" [shape=hexagon, fillcolor="{colour(t)}", label="{s}\\ntrust={t:.2f}"];')
        dot.append("}")
        st.graphviz_chart("\n".join(dot), width="stretch")
        st.caption("Couleur des capteurs : vert = fiable, orange = douteux, rouge = peu fiable.")

# ---------------------------------------------------------------- décisions
with tab_ev:
    strat = st.radio("Stratégie", STRATEGIES, horizontal=True,
                     format_func=lambda s: "Naïve (tout alerter)" if s == "naive"
                     else "Sensible à la confiance")
    ev = ev_by[strat]
    flt = st.multiselect("Statut", ["confirmed", "pending_verification"],
                         default=["confirmed", "pending_verification"])
    view = ev[ev["status"].isin(flt)][["appliance", "sensor", "start", "end", "duration_h",
                                       "severity", "confidence", "status", "is_real",
                                       "recommendation"]]
    st.dataframe(view.rename(columns={"is_real": "vraie anomalie ?"}),
                 hide_index=True, width="stretch")
    st.caption("La colonne « vraie anomalie ? » vient de la vérité terrain de la simulation : "
               "les agents n'y ont jamais accès.")

# ---------------------------------------------------------------- expérience
with tab_exp:
    st.subheader("Ce scénario")
    table = pd.DataFrame(summ)
    st.dataframe(table, width="stretch")
    chart_df = pd.DataFrame({
        "stratégie": ["naive", "trust_aware"] * 2,
        "type": ["alertes vraies"] * 2 + ["fausses alarmes"] * 2,
        "n": [summ[s]["alerts_sent"] - summ[s]["false_alarms"] for s in STRATEGIES]
             + [summ[s]["false_alarms"] for s in STRATEGIES]})
    st.altair_chart(alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("stratégie:N", title=None), y=alt.Y("n:Q", title="Alertes envoyées"),
        color=alt.Color("type:N", scale=alt.Scale(domain=["alertes vraies", "fausses alarmes"],
                                                   range=["#54a24b", "#e45756"]))).properties(height=250),
        width="stretch")

    st.subheader("Robustesse : moyenne sur plusieurs graines aléatoires")
    n = st.slider("Nombre de graines", 5, 50, 20, 5)
    if st.checkbox("Lancer l'expérience répétée"):
        many = robustness(tuple(sorted(faulty)), fault_rate, threshold, n)
        cols = ["alerts_sent", "false_alarms", "precision", "real_anomalies_alerted",
                "real_events_deferred"]
        agg = many.groupby("strategy")[cols].agg(["mean", "std"]).round(2)
        st.dataframe(agg, width="stretch")
        st.caption("Compromis : la stratégie sensible à la confiance supprime presque toutes "
                   "les fausses alarmes, mais retarde certaines vraies anomalies "
                   "(real_events_deferred).")

# ---------------------------------------------------------------- assistant
with tab_chat:
    st.caption(f"Mode : **{agent.mode}** — " + (
        f"{agent.label} : le LLM traduit la question en SPARQL puis rédige la réponse à partir des résultats."
        if agent.mode == "llm" else
        "sans clé API, des règles simples choisissent une requête prête à l'emploi "
        "(anomalies, capteurs, consommation)."))
    examples = ["(ma propre question)", "Quelles anomalies sont confirmées ?",
                "Quels capteurs sont peu fiables ?", "Quel appareil consomme le plus ?"]
    pick = st.selectbox("Exemples", examples)
    typed = st.text_input("Votre question", placeholder="Ex. : Quels événements viennent du capteur Sensor02 ?")
    question = typed.strip() if pick == examples[0] else pick
    c1, c2 = st.columns(2)
    if c1.button("Poser la question") and question:
        with st.spinner("Les agents interrogent le graphe…"):
            st.session_state["answer"] = agent.ask(question, kg)
    if c2.button("Générer un rapport"):
        with st.spinner("Rédaction du rapport…"):
            st.session_state["report"] = agent.report(ev_t, trust, all_appliances=[n for n,_,_ in info])

    ans = st.session_state.get("answer")
    if ans:
        st.markdown(f"**Question :** {ans.question}")
        if ans.error:
            st.error(ans.error)
        else:
            st.success(ans.text)
            with st.expander("Voir la requête SPARQL et les données brutes"):
                st.code(ans.sparql, language="sparql")
                st.dataframe(pd.DataFrame(ans.rows, columns=ans.columns), hide_index=True,
                             width="stretch")
    if st.session_state.get("report"):
        st.subheader("Rapport")
        st.info(st.session_state["report"])

# ---------------------------------------------------------------- SPARQL
with tab_sparql:
    st.caption(f"Graphe courant : {len(kg)} triples (ontologie + bâtiment + mesures simulées + "
               f"conclusions des agents).")
    qdir = ROOT / "queries"
    files = sorted(qdir.glob("*.rq")) if qdir.exists() else []
    choice = st.selectbox("Requête prête à l'emploi", ["(écrire la mienne)"] + [f.name for f in files])
    default = ("PREFIX sb: <http://smartbuilding.org#>\nSELECT ?appliance ?status ?confidence\n"
               "WHERE { ?e a sb:AnomalyEvent ; sb:affects ?appliance ; sb:status ?status ;\n"
               "        sb:confidence ?confidence . }\nLIMIT 10")
    text = (qdir / choice).read_text(encoding="utf-8") if choice != "(écrire la mienne)" else default
    query = st.text_area("Requête SPARQL", text, height=200, key=f"q_{choice}")
    if st.button("Exécuter"):
        try:
            res = kg.query(query)
            out = pd.DataFrame([[local(v) if v is not None else "-" for v in row] for row in res],
                               columns=[str(v) for v in res.vars])
            st.dataframe(out, hide_index=True, width="stretch")
            st.caption(f"{len(out)} ligne(s)")
        except Exception as e:
            st.error(f"Erreur SPARQL : {e}")

# ---------------------------------------------------------------- architecture
with tab_about:
    st.graphviz_chart("""digraph A { rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica];
      KG [label="Knowledge Graph\\n(RDF / OWL)", shape=cylinder];
      D [label="DetectorAgent\\nIsolation Forest"]; T [label="TrustAgent\\nscore de confiance"];
      O [label="Orchestrator\\nagir / vérifier"]; R [label="RecommenderAgent\\naction proposée"];
      KG -> D [label="SPARQL"]; D -> T; T -> O; D -> O; O -> R; R -> KG [label="triples + PROV-O"]; }""",
                        width="stretch")
    st.markdown("""
**Limites assumées** : données simulées (la vérité terrain provient du simulateur) ;
hypothèse du TrustAgent simple (défaut = événement court) ; le compromis fausses alarmes /
anomalies retardées est un résultat, pas un défaut à cacher.
""")
