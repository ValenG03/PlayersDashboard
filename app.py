import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

st.set_page_config(
    page_title="2026 World Cup Player Dashboard",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f8fbff, #eef5ff);
}
h1, h2, h3 {
    color: #12355b;
}
.stMetric {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}
div.stButton > button {
    background-color: #12355b;
    color: white;
    border-radius: 12px;
    padding: 10px 22px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚽ 2026 World Cup Player Performance Dashboard")
st.caption("Interactive scouting dashboard using Python, Pandas, Plotly, Seaborn and clustering")

@st.cache_data
def load_data():
    return pd.read_csv("players_data-2025_2026.csv")

df = load_data()

features = ["Gls", "Ast", "Sh", "SoT", "Crs", "Int", "TklW", "CrdY", "MP", "Min"]

feature_explanations = {
    "Gls": "Goals scored",
    "Ast": "Assists made",
    "Sh": "Total shots",
    "SoT": "Shots on target",
    "Crs": "Crosses",
    "Int": "Interceptions",
    "TklW": "Tackles won",
    "CrdY": "Yellow cards",
    "MP": "Matches played",
    "Min": "Minutes played"
}

with st.expander("What do these stats mean?"):
    for stat in features:
        st.write(f"**{stat}**: {feature_explanations[stat]}")


df = df.dropna(subset=features)

st.sidebar.header("🎛️ Filters")

teams = st.sidebar.multiselect(
    "Select teams",
    sorted(df["Squad"].unique()),
    default=sorted(df["Squad"].unique())[:10],
    key="team_filter"
)

positions = st.sidebar.multiselect(
    "Select positions",
    sorted(df["Pos"].unique()),
    default=sorted(df["Pos"].unique()),
    key="position_filter"
)

cluster_method = st.sidebar.radio(
    "Choose clustering method",
    ["KMeans", "Agglomerative"],
    key="cluster_method_radio"
)

k = st.sidebar.slider(
    "Number of clusters",
    2,
    6,
    4,
    key="cluster_slider"
)


df = df[
    (df["Squad"].isin(teams)) &
    (df["Pos"].isin(positions))
]

if len(df) < k:
    st.warning("Not enough players for the selected number of clusters. Choose more teams or fewer clusters.")
    st.stop()

X = StandardScaler().fit_transform(df[features])

if cluster_method == "KMeans":
    df["Cluster"] = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    ).fit_predict(X)
else:
    df["Cluster"] = AgglomerativeClustering(
        n_clusters=k
    ).fit_predict(X)

style_names = {
    0: "Finishers",
    1: "Creators",
    2: "Defensive Workers",
    3: "Wide Progressors",
    4: "Balanced Players",
    5: "High Activity Players"
}

df["Playing Style"] = df["Cluster"].map(style_names)

colors = sns.color_palette("husl", k).as_hex()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Players", len(df))
c2.metric("Teams", df["Squad"].nunique())
c3.metric("Positions", df["Pos"].nunique())
c4.metric("Clusters", k)

st.subheader("Player Style Map")

fig = px.scatter(
    df,
    x="Sh",
    y="Ast",
    size="Gls",
    color="Playing Style",
    hover_name="Player",
    hover_data=["Nation", "Squad", "Pos", "Gls", "Ast", "Sh", "SoT", "Min"],
    color_discrete_sequence=colors,
    title="Players clustered by attacking and creative profile"
)

fig.update_layout(
    template="plotly_white",
    height=600,
    title_font_size=24
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🏆 Top 10 Players by Category")

category = st.selectbox(
    "Choose a statistic",
    features,
    key="top10_category_selectbox"
)

top10 = df.sort_values(category, ascending=False).head(10)

top_fig = px.bar(
    top10,
    x=category,
    y="Player",
    color=category,
    orientation="h",
    hover_data=["Squad", "Nation", "Pos"],
    title=f"Top 10 Players by {category}",
    color_continuous_scale="Turbo"
)

top_fig.update_layout(
    template="plotly_white",
    yaxis={"categoryorder": "total ascending"},
    height=500
)

st.plotly_chart(top_fig, use_container_width=True)


st.subheader("🥧 Team Goal & Assist Share")

goal_data = df.groupby("Squad")["Gls"].sum().sort_values(ascending=False).head(10).reset_index()
assist_data = df.groupby("Squad")["Ast"].sum().sort_values(ascending=False).head(10).reset_index()

col1, col2 = st.columns(2)

with col1:
    goal_pie = px.pie(
        goal_data,
        names="Squad",
        values="Gls",
        title="Top Teams by Goals",
        hole=0.35,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(goal_pie, use_container_width=True)

with col2:
    assist_pie = px.pie(
        assist_data,
        names="Squad",
        values="Ast",
        title="Top Teams by Assists",
        hole=0.35,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(assist_pie, use_container_width=True)


st.subheader("📊 Average Stats by Playing Style")

summary = (
    df.groupby("Playing Style")[features]
    .mean()
    .round(2)
    .reset_index()
)

bar = px.bar(
    summary,
    x="Playing Style",
    y=features,
    barmode="group",
    title="Cluster Profile Comparison",
    color_discrete_sequence=px.colors.qualitative.Bold
)

bar.update_layout(
    template="plotly_white",
    height=500
)

st.plotly_chart(bar, use_container_width=True)

st.subheader("🎯 Player Explorer")

player = st.selectbox(
    "Choose a player",
    sorted(df["Player"].unique()),
    key="player_explorer_selectbox"
)

player_data = df[df["Player"] == player].iloc[0]

st.success(
    f"{player} plays for {player_data['Squad']} and belongs to the style: {player_data['Playing Style']}."
)

if st.button("Celebrate this player 🎉", key="celebrate_button"):
    st.balloons()

st.subheader("🔎 Full Player Table")

st.dataframe(
    df[["Player", "Nation", "Squad", "Pos", "Playing Style"] + features]
    .sort_values("Gls", ascending=False),
    use_container_width=True
)
