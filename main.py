"""
🐕 Plateforme d'analyse statistique canine
Application Streamlit tout-en-un : descriptives, corrélations, ACP, ACM,
FAMD, classification, prédiction, interprétation IA (DeepSeek / Kimi),
génération de rapport.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA as SkPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from scipy.cluster.hierarchy import dendrogram, linkage
import prince

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dogs Stats Platform",
    page_icon="🐕",
    layout="wide",
)

NUMERIC_COLS = ["HW", "HR", "BL", "HL", "HEW", "ML", "HG",
                "EL", "CG", "WG", "AG", "NL", "LW"]

AI_PROVIDERS = {
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    "kimi":     {"base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
}


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "df": None,
        "file_name": None,
        "ai_provider": "deepseek",
        "ai_api_key": "",
        "report_sections": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), sep=";", decimal=",")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[c for c in NUMERIC_COLS if c in df.columns])
    return df


# ----------------------------------------------------------------------------
# STATS ENGINE
# ----------------------------------------------------------------------------
def descriptive_stats(df, group_col=None):
    if group_col and group_col in df.columns:
        return df.groupby(group_col).describe().T
    return df.describe().T


def normality_tests(df, cols):
    rows = []
    for col in cols:
        data = df[col].dropna()
        if len(data) < 3:
            continue
        stat, p = stats.shapiro(data)
        rows.append({"Variable": col, "W": round(stat, 4),
                     "p-value": round(p, 6),
                     "Normal (α=0.05)": "Oui" if p > 0.05 else "Non"})
    return pd.DataFrame(rows)


def levene_test(df, cols, group_col):
    groups = [g[col].dropna().values for _, g in df.groupby(group_col)
              for col in [cols[0]]] if False else None
    rows = []
    for col in cols:
        samples = [g[col].dropna().values for _, g in df.groupby(group_col)]
        if len(samples) >= 2 and all(len(s) > 1 for s in samples):
            stat, p = stats.levene(*samples)
            rows.append({"Variable": col, "Levene": round(stat, 4),
                         "p-value": round(p, 6),
                         "Variances égales": "Oui" if p > 0.05 else "Non"})
    return pd.DataFrame(rows)


def anova_or_kruskal(df, cols, group_col):
    rows = []
    for col in cols:
        groups = [g[col].dropna().values for _, g in df.groupby(group_col)]
        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            continue
        # Normalité par groupe
        normal = all(stats.shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3)
        if normal:
            stat, p = stats.f_oneway(*groups)
            test = "ANOVA"
        else:
            stat, p = stats.kruskal(*groups)
            test = "Kruskal-Wallis"
        rows.append({"Variable": col, "Test": test,
                     "Statistique": round(stat, 4),
                     "p-value": round(p, 6),
                     "Différence significative": "Oui" if p < 0.05 else "Non"})
    return pd.DataFrame(rows)


def correlation_matrix(df, cols, method="pearson"):
    return df[cols].corr(method=method)


def run_pca(df, cols, n_components=5):
    X = StandardScaler().fit_transform(df[cols])
    pca = prince.PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    return pca


def run_mca(df, cat_cols, n_components=5):
    mca = prince.MCA(n_components=n_components, random_state=42)
    mca.fit(df[cat_cols].astype(str))
    return mca


def run_famd(df, num_cols, cat_cols, n_components=5):
    famd = prince.FAMD(n_components=n_components, random_state=42)
    famd.fit(df[num_cols + cat_cols])
    return famd


# ----------------------------------------------------------------------------
# AI INTERPRETER
# ----------------------------------------------------------------------------
def ai_interpret(context: str, prompt: str) -> str:
    provider = st.session_state.ai_provider
    api_key = st.session_state.ai_api_key or st.secrets.get(
        "DEEPSEEK_API_KEY" if provider == "deepseek" else "KIMI_API_KEY", ""
    )
    if not api_key:
        return "⚠️ Aucune clé API configurée. Renseignez-la dans la barre latérale."
    try:
        from openai import OpenAI
        cfg = AI_PROVIDERS[provider]
        client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
        system = (
            "Tu es un statisticien expert en morphométrie animale. "
            "Interprète les résultats fournis de manière rigoureuse, en citant "
            "les valeurs numériques pertinentes. Réponds en français, en 3 à 5 "
            "paragraphes structurés."
        )
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Contexte :\n{context}\n\nDemande : {prompt}"},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur API : {e}"


def add_to_report(title: str, content: str):
    st.session_state.report_sections.append({"title": title, "content": content})


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🐕 Dogs Stats")
    st.markdown("**Plateforme d'analyse morphométrique canine**")
    st.divider()

    st.header("📁 Données")
    uploaded = st.file_uploader("Fichier CSV (`;` séparateur)", type=["csv"])
    if uploaded is not None:
        try:
            df = load_and_clean(uploaded.getvalue())
            st.session_state.df = df
            st.session_state.file_name = uploaded.name
            st.success(f"✅ {df.shape[0]} × {df.shape[1]}")
        except Exception as e:
            st.error(f"Erreur : {e}")

    st.divider()
    st.header("🤖 IA")
    st.session_state.ai_provider = st.selectbox(
        "Fournisseur", list(AI_PROVIDERS.keys())
    )
    st.session_state.ai_api_key = st.text_input(
        "Clé API", type="password",
        help="Stockée en session uniquement. Sur Streamlit Cloud, utilisez st.secrets."
    )

    st.divider()
    st.caption("v1.0 — Streamlit + prince + plotly")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
st.title("🐕 Plateforme d'analyse statistique canine")

df = st.session_state.df
if df is None:
    st.info("👉 Chargez un fichier CSV dans la barre latérale pour commencer.")
    st.markdown("""
    ### Fonctionnalités
    - **Descriptives** : moyennes, écarts-types, tests de normalité, outliers
    - **Corrélations** : Pearson, Spearman, Kendall, heatmap
    - **ACP / ACM / FAMD** : analyses factorielles complètes
    - **Classification** : CAH, k-means, DBSCAN
    - **Prédiction** : LDA, Random Forest, importance des variables
    - **Interprétation IA** : DeepSeek ou Kimi
    - **Rapport** : export Markdown avec tous les résultats
    """)
    st.stop()

numeric_in_df = [c for c in NUMERIC_COLS if c in df.columns]
cat_in_df = [c for c in ["SEX", "BREED"] if c in df.columns]

tabs = st.tabs([
    "📊 Descriptives", "🔥 Corrélations", "🧬 ACP", "🔗 ACM", "🧪 FAMD",
    "🌳 Classification", "🤖 Prédiction", "📝 Rapport",
])


# ---------- TAB 1 : DESCRIPTIVES ----------
with tabs[0]:
    st.header("📊 Statistiques descriptives")
    st.dataframe(df.head(20), use_container_width=True)

    group_col = st.selectbox("Grouper par", ["Aucun"] + cat_in_df, key="desc_grp")
    gc = None if group_col == "Aucun" else group_col

    st.subheader("Statistiques")
    desc = descriptive_stats(df[numeric_in_df + ([gc] if gc else [])], gc)
    st.dataframe(desc, use_container_width=True)

    st.subheader("Tests de normalité (Shapiro-Wilk)")
    norm = normality_tests(df, numeric_in_df)
    st.dataframe(norm, use_container_width=True)

    st.subheader("Boxplots")
    var_box = st.selectbox("Variable", numeric_in_df, key="box_var")
    if gc:
        fig = px.box(df, x=gc, y=var_box, color=gc, points="all")
    else:
        fig = px.box(df, y=var_box, points="all")
    st.plotly_chart(fig, use_container_width=True)

    if gc:
        st.subheader("Tests de comparaison de groupes")
        anova = anova_or_kruskal(df, numeric_in_df, gc)
        st.dataframe(anova, use_container_width=True)

        lev = levene_test(df, numeric_in_df, gc)
        st.subheader("Test de Levene (homogénéité des variances)")
        st.dataframe(lev, use_container_width=True)

    if st.button("➕ Ajouter au rapport", key="add_desc"):
        add_to_report("Statistiques descriptives",
                      desc.head(30).to_markdown() + "\n\n" +
                      norm.to_markdown())
        st.success("Ajouté au rapport.")

    if st.button("🤖 Interpréter", key="ai_desc"):
        ctx = desc.head(30).to_string() + "\n\n" + norm.to_string()
        st.markdown(ai_interpret(ctx, "Interprète ces statistiques descriptives."))


# ---------- TAB 2 : CORRELATIONS ----------
with tabs[1]:
    st.header("🔥 Analyse des corrélations")
    method = st.radio("Méthode", ["pearson", "spearman", "kendall"], horizontal=True)
    corr = correlation_matrix(df, numeric_in_df, method=method)

    st.subheader("Matrice de corrélation")
    st.dataframe(corr.round(3), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, ax=ax)
    st.pyplot(fig)

    # Top corrélations
    st.subheader("Corrélations les plus fortes (|r| > 0.5)")
    pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                 .stack().reset_index())
    pairs.columns = ["Var1", "Var2", "r"]
    strong = pairs[pairs["r"].abs() > 0.5].sort_values("r", key=abs, ascending=False)
    st.dataframe(strong, use_container_width=True)

    if st.button("➕ Ajouter au rapport", key="add_corr"):
        add_to_report("Corrélations",
                      f"Méthode : {method}\n\n" + strong.to_markdown(index=False))
        st.success("Ajouté.")

    if st.button("🤖 Interpréter", key="ai_corr"):
        ctx = strong.head(15).to_string()
        st.markdown(ai_interpret(ctx, "Interprète les corrélations les plus fortes."))


# ---------- TAB 3 : ACP ----------
with tabs[2]:
    st.header("🧬 Analyse en Composantes Principales (ACP)")
    selected = st.multiselect("Variables actives", numeric_in_df,
                              default=numeric_in_df[:8], key="pca_vars")
    if len(selected) >= 2:
        pca = run_pca(df, selected, n_components=5)
        eig = np.array(pca.eigenvalues_[:5])
        inertia = np.array(pca.explained_inertia_[:5]) * 100

        st.subheader("Valeurs propres et variance expliquée")
        st.dataframe(pd.DataFrame({
            "Axe": [f"F{i+1}" for i in range(len(eig))],
            "Valeur propre": eig.round(4),
            "Variance expliquée (%)": inertia.round(2),
            "Cumulée (%)": np.cumsum(inertia).round(2),
        }))

        st.subheader("Cercle des corrélations (F1 × F2)")
        loadings = pca.column_coordinates_
        fig = px.scatter(loadings, x=0, y=1, text=loadings.index)
        fig.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1,
                      line=dict(color="gray", dash="dot"))
        fig.update_traces(textposition="top center")
        fig.update_layout(xaxis_title="F1", yaxis_title="F2")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Plan factoriel des individus")
        row_coords = pca.row_coordinates(df[selected].values)
        row_coords.columns = [f"F{i+1}" for i in range(row_coords.shape[1])]
        if "BREED" in df.columns:
            row_coords["BREED"] = df["BREED"].values
            fig2 = px.scatter(row_coords, x="F1", y="F2", color="BREED")
        else:
            fig2 = px.scatter(row_coords, x="F1", y="F2")
        st.plotly_chart(fig2, use_container_width=True)

        if st.button("➕ Ajouter au rapport", key="add_pca"):
            add_to_report("ACP", loadings.to_markdown())
            st.success("Ajouté.")

        if st.button("🤖 Interpréter", key="ai_pca"):
            ctx = f"Matrice de saturations :\n{loadings.to_string()}\n\nVariance : {inertia}"
            st.markdown(ai_interpret(ctx, "Explique les axes F1 et F2."))


# ---------- TAB 4 : ACM ----------
with tabs[3]:
    st.header("🔗 Analyse des Correspondances Multiples (ACM)")
    if len(cat_in_df) < 2:
        st.warning("Il faut au moins 2 variables qualitatives (SEX, BREED).")
    else:
        cat_sel = st.multiselect("Variables actives", cat_in_df,
                                 default=cat_in_df, key="mca_vars")
        if len(cat_sel) >= 2:
            mca = run_mca(df, cat_sel)
            eig = np.array(mca.eigenvalues_[:5])

            st.subheader("Valeurs propres")
            st.dataframe(pd.DataFrame({
                "Axe": [f"F{i+1}" for i in range(len(eig))],
                "Valeur propre": eig.round(4),
                "% inertie": (100 * eig / eig.sum()).round(2),
            }))

            st.subheader("Carte des modalités (F1 × F2)")
            col_coords = mca.column_coordinates(df[cat_sel].astype(str))
            col_coords.columns = [f"F{i+1}" for i in range(col_coords.shape[1])]
            fig = px.scatter(col_coords, x="F1", y="F2", text=col_coords.index)
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)

            if st.button("➕ Ajouter au rapport", key="add_mca"):
                add_to_report("ACM", col_coords.to_markdown())
                st.success("Ajouté.")

            if st.button("🤖 Interpréter", key="ai_mca"):
                ctx = col_coords.to_string()
                st.markdown(ai_interpret(ctx, "Interprète les modalités sur F1 et F2."))


# ---------- TAB 5 : FAMD ----------
with tabs[4]:
    st.header("🧪 FAMD (analyse factorielle de données mixtes)")
    num_sel = st.multiselect("Variables quantitatives", numeric_in_df,
                             default=numeric_in_df[:6], key="famd_num")
    cat_sel2 = st.multiselect("Variables qualitatives", cat_in_df,
                              default=cat_in_df, key="famd_cat")
    if num_sel and cat_sel2:
        famd = run_famd(df, num_sel, cat_sel2)
        eig = np.array(famd.eigenvalues_[:5])
        st.dataframe(pd.DataFrame({
            "Axe": [f"F{i+1}" for i in range(len(eig))],
            "Valeur propre": eig.round(4),
            "% inertie": (100 * eig / eig.sum()).round(2),
        }))

        col_coords = famd.column_coordinates(df[num_sel + cat_sel2])
        col_coords.columns = [f"F{i+1}" for i in range(col_coords.shape[1])]
        fig = px.scatter(col_coords, x="F1", y="F2", text=col_coords.index)
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

        if st.button("➕ Ajouter au rapport", key="add_famd"):
            add_to_report("FAMD", col_coords.to_markdown())
            st.success("Ajouté.")


# ---------- TAB 6 : CLASSIFICATION ----------
with tabs[5]:
    st.header("🌳 Classification non supervisée")
    method_cl = st.selectbox("Méthode",
                             ["CAH (Ward)", "k-means", "DBSCAN"], key="clust_method")
    X = StandardScaler().fit_transform(df[numeric_in_df])

    if method_cl == "CAH (Ward)":
        st.subheader("Dendrogramme")
        Z = linkage(X, method="ward")
        fig, ax = plt.subplots(figsize=(12, 5))
        dendrogram(Z, ax=ax, no_labels=True)
        st.pyplot(fig)

        k = st.slider("Nombre de clusters", 2, 8, 3)
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)

    elif method_cl == "k-means":
        k = st.slider("Nombre de clusters", 2, 8, 3)
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)

        st.subheader("Méthode du coude")
        inertias = [KMeans(n_clusters=i, n_init=10, random_state=42)
                    .fit(X).inertia_ for i in range(2, 9)]
        st.line_chart(pd.DataFrame({"k": range(2, 9), "inertie": inertias}).set_index("k"))

    else:
        eps = st.slider("eps", 0.5, 5.0, 1.5, 0.1)
        min_s = st.slider("min_samples", 2, 20, 5)
        labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X)

    st.subheader("Visualisation (projection ACP)")
    pca2 = SkPCA(n_components=2).fit_transform(X)
    proj = pd.DataFrame(pca2, columns=["PC1", "PC2"])
    proj["Cluster"] = labels.astype(str)
    fig = px.scatter(proj, x="PC1", y="PC2", color="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Effectifs par cluster")
    st.dataframe(pd.Series(labels).value_counts().rename("Effectif"))

    if st.button("➕ Ajouter au rapport", key="add_clust"):
        add_to_report("Classification",
                      f"Méthode : {method_cl}\n\nEffectifs :\n" +
                      pd.Series(labels).value_counts().to_markdown())
        st.success("Ajouté.")


# ---------- TAB 7 : PREDICTION ----------
with tabs[6]:
    st.header("🤖 Prédiction supervisée")
    target = st.selectbox("Variable cible", cat_in_df, key="pred_target")
    if target and len(numeric_in_df) >= 2:
        X = df[numeric_in_df].values
        y = df[target].astype(str).values

        model_name = st.selectbox("Modèle", ["LDA", "Random Forest"], key="pred_model")

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                                   random_state=42, stratify=y)

        if model_name == "LDA":
            model = LinearDiscriminantAnalysis()
        else:
            model = RandomForestClassifier(n_estimators=200, random_state=42)

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = (y_pred == y_te).mean()
        st.metric("Accuracy (test 30 %)", f"{acc:.3f}")

        st.subheader("Matrice de confusion")
        cm = confusion_matrix(y_te, y_pred, labels=np.unique(y))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax)
        st.pyplot(fig)

        st.subheader("Rapport de classification")
        st.text(classification_report(y_te, y_pred))

        if model_name == "Random Forest":
            st.subheader("Importance des variables")
            imp = pd.DataFrame({"Variable": numeric_in_df,
                                "Importance": model.feature_importances_}
                               ).sort_values("Importance", ascending=False)
            st.dataframe(imp, use_container_width=True)
            st.bar_chart(imp.set_index("Variable"))

        if st.button("➕ Ajouter au rapport", key="add_pred"):
            add_to_report("Prédiction",
                          f"Cible : {target}, modèle : {model_name}, accuracy : {acc:.3f}\n\n"
                          + classification_report(y_te, y_pred))
            st.success("Ajouté.")


# ---------- TAB 8 : RAPPORT ----------
with tabs[7]:
    st.header("📝 Rapport d'analyse")
    sections = st.session_state.report_sections
    if not sections:
        st.info("Aucune section ajoutée pour le moment. Utilisez les boutons "
                "« ➕ Ajouter au rapport » dans les autres onglets.")
    else:
        st.success(f"{len(sections)} section(s) dans le rapport.")
        full_md = "# Rapport d'analyse statistique canine\n\n"
        for i, sec in enumerate(sections, 1):
            with st.expander(f"{i}. {sec['title']}", expanded=False):
                st.markdown(sec["content"])
            full_md += f"## {i}. {sec['title']}\n\n{sec['content']}\n\n"

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Télécharger Markdown",
                               full_md, file_name="rapport_analyse.md",
                               mime="text/markdown")
        with col2:
            if st.button("🗑️ Vider le rapport"):
                st.session_state.report_sections = []
                st.rerun()

        st.divider()
        st.subheader("🤖 Génération d'une synthèse narrative par l'IA")
        if st.button("Générer la synthèse complète"):
            ctx = "\n\n".join(f"### {s['title']}\n{s['content']}"
                              for s in sections)[:12000]
            st.markdown(ai_interpret(
                ctx,
                "Rédige une synthèse narrative complète et structurée de ces "
                "résultats statistiques, avec une introduction, une discussion "
                "et une conclusion en français."
            ))
