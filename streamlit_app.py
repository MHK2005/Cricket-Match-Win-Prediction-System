import streamlit as st
import pandas as pd
import pickle
import random
from datetime import datetime
import streamlit.components.v1 as components

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Cricket Match Win Predictor",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 Cricket Match Win Predictor")
st.write("Predict outcomes of cricket matches")

# --------------------------------------------------
# LOAD MODELS & ENCODERS
# --------------------------------------------------
@st.cache_resource
def load_assets():
    with open("cricket_prediction_model.pkl", "rb") as f:
        return pickle.load(f)

data = load_assets()

model_t20 = data["model_t20"]
model_odi = data["model_odi"]
model_test = data["model_test"]

team_encoder = data["encoder"]
venue_encoder = data["venue_encoder"]

teams = team_encoder.classes_
venues = venue_encoder.classes_

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
def init_state():
    if "teams_confirmed" not in st.session_state:
        st.session_state.teams_confirmed = False
    
    if "match_randomized" not in st.session_state:
        st.session_state.match_randomized = False

    defaults = {
        "toss_winner": None,
        "toss_decision": None,
        "pitch": None,
        "venue": None,
        "venue_avg": None,
        "t1_score": None,
        "t2_score": None,
        "t1_wkts": None,
        "t2_wkts": None
    }
    
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# --------------------------------------------------
# RANDOM MATCH SCENARIO
# --------------------------------------------------
def randomize_match():
    if not st.session_state.get("teams_confirmed"):
        return

    if not team1 or not team2 or team1 == team2:
        return

    st.session_state.toss_winner = random.choice([team1, team2])
    st.session_state.toss_decision = random.choice(["Batting", "Bowling"])
    st.session_state.pitch = random.choice(
        ["Balanced", "Batting Friendly", "Bowling Friendly"]
    )

    if match_format == "T20":
        base = random.randint(150, 190)
    elif match_format == "ODI":
        base = random.randint(260, 310)
    else:
        base = random.randint(380, 450)

    st.session_state.venue_avg = base
    st.session_state.t1_score = random.randint(base - 30, base + 20)
    st.session_state.t2_score = random.randint(base - 40, base + 10)
    st.session_state.t1_wkts = random.randint(2, 8)
    st.session_state.t2_wkts = random.randint(3, 10)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    match_format = st.selectbox(
        "Match Format",
        ["T20", "ODI", "TEST"],
        index=1,
        placeholder="Select Format...",
        on_change=randomize_match
    )

    if st.button("🔄 Reset All", key="reset_btn"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.team1_select = None
        st.session_state.team2_select = None
        st.rerun()

# --------------------------------------------------
# STEP 1: TEAMS
# --------------------------------------------------
st.divider()
st.subheader("1️⃣ Teams")

c1, c2 = st.columns(2)

team1 = c1.selectbox(
    "Team 1",
    teams,
    index=None,
    placeholder="Select Team 1",
    key="team1_select"
)

team2 = c2.selectbox(
    "Team 2",
    teams,
    index=None,
    placeholder="Select Team 2",
    key="team2_select"
)

if not team1 or not team2:
    st.warning("👆 Please select teams first!")
    st.stop()

if team1 == team2:
    st.error("⚠️ Please select different teams!")
    st.stop()

st.session_state.teams_confirmed = True

st.success(f"✅ Teams selected: **{team1}** vs **{team2}**")

if not st.session_state.teams_confirmed:
    st.stop()

# --------------------------------------------------
# STEP 2: VENUE & TOSS
# --------------------------------------------------
st.divider()
st.subheader("2️⃣ Venue & Toss")

c1, c2, c3 = st.columns(3)

venue = c1.selectbox("Venue", venues, index=None, placeholder="Select Venue")
toss_winner = c2.selectbox("Toss Winner", [team1, team2], index=None, placeholder="Select Toss Winner")
toss_decision = c3.selectbox("Toss Decision", ["Batting", "Bowling"], index=None, placeholder="Select Toss Decision")

pitch_condition = st.selectbox(
    "Pitch Condition",
    ["Balanced", "Batting Friendly", "Bowling Friendly"],
    index=None,
    placeholder="Select Pitch Condition"
)

if (venue and toss_winner and toss_decision and pitch_condition):
    st.success(f"✅ Venue & Toss set!  \n\nVenue: **{venue}**  \nToss: **{toss_winner}** won the toss and chose to **{"Bat" if toss_decision == "Batting" else "Bowl"}** first")
else:    
    st.warning("👆 Please complete venue and toss details")


if not (venue and toss_winner and toss_decision and pitch_condition):
    st.stop()

if not st.session_state.match_randomized:
    randomize_match()
    st.session_state.match_randomized = True

# --------------------------------------------------
# STEP 3: LIVE MATCH INPUT
# --------------------------------------------------
st.divider()
st.subheader("3️⃣ Match Details")

# Decide batting order from toss
batting_first = (
    toss_winner if toss_decision == "Batting"
    else team1 if toss_winner == team2
    else team2
)

bowling_first = team2 if batting_first == team1 else team1

if match_format == "TEST":
    st.info("👇 Enter combined match progress")

    c1, c2 = st.columns(2)
    t1_runs = c1.number_input(f"{team1} Runs", 0, 800, st.session_state.t1_score)
    t1_wkts = c2.number_input(f"{team1} Wickets", 0, 20, st.session_state.t1_wkts)

    c1, c2 = st.columns(2)
    t2_runs = c1.number_input(f"{team2} Runs", 0, 800, st.session_state.t2_score)
    t2_wkts = c2.number_input(f"{team2} Wickets", 0, 20, st.session_state.t2_wkts)

else:
    innings = st.radio("Innings", ["1st Innings", "2nd Innings"], horizontal=True)

    if innings == "1st Innings":
        runs = st.number_input("Current Runs", 0, 400, st.session_state.t1_score)
        wkts = st.number_input("Wickets Down", 0, 10, st.session_state.t1_wkts)
        overs = st.slider("Overs Played", 1, 50 if match_format == "ODI" else 20)

        run_rate = runs / overs
        max_overs = 50 if match_format == "ODI" else 20
        projected = int(run_rate * max_overs)

        st.metric("Projected Score", projected)

        t1_runs, t1_wkts = projected, min(10, wkts + 2)
        t2_runs, t2_wkts = st.session_state.venue_avg, 10

    else:
        target = st.number_input("Target", 50, 500, st.session_state.venue_avg)
        runs = st.number_input("Current Runs", 0, target, st.session_state.t2_score)
        wkts = st.number_input("Wickets Down", 0, 10, st.session_state.t2_wkts)
        overs = st.slider("Overs Played", 1, 50 if match_format == "ODI" else 20)

        run_rate = runs / overs
        max_overs = 50 if match_format == "ODI" else 20
        projected = int(run_rate * max_overs)

        t1_runs, t1_wkts = target, 8
        t2_runs, t2_wkts = projected, wkts
    
    if (t1_runs is not None and t2_runs is not None):
        st.success("✅ Match is ready for prediction")
    else:    
        st.warning("👆 Please enter match details first")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🚀 Submit", type="primary"):

    # Encoding
    t1_enc = team_encoder.transform([team1])[0]
    t2_enc = team_encoder.transform([team2])[0]
    venue_enc = venue_encoder.transform([venue])[0]

    toss_enc = team_encoder.transform([toss_winner])[0]
    toss_dec_enc = 1 if toss_decision == "Batting" else 0

    pitch_bat = 1 if pitch_condition == "Batting Friendly" else 0
    pitch_bowl = 1 if pitch_condition == "Bowling Friendly" else 0

    month = datetime.now().month

    # Build all possible features (safe superset)
    feature_dict = {
        # Venue & pitch
        "Venue": venue_enc,
        "Venue Average Score": st.session_state.venue_avg,
        "Pitch_Batting Friendly": 1 if pitch_condition == "Batting Friendly" else 0,
        "Pitch_Bowling Friendly": 1 if pitch_condition == "Bowling Friendly" else 0,

        # Toss
        "Toss Decision Encoded": toss_dec_enc,
        "Toss Winner Encoded": toss_enc,

        # Teams
        "Team1 Encoded": t1_enc,
        "Team2 Encoded": t2_enc,

        # Match timing
        "Match Month": month,
        "Match Start Month": month,
        "Match End Month": month,
        "Diff Month": 0,

        # Performance
        "Team1 Batting Average": t1_runs / max(1, t1_wkts),
        "Team2 Batting Average": t2_runs / max(1, t2_wkts),
    }

    if match_format == "TEST":
        model = model_test
        required_features = model.feature_names_in_
        input_df = pd.DataFrame(
            [{k: feature_dict[k] for k in required_features}]
        )
    else:
        model = model_t20 if match_format == "T20" else model_odi
        required_features = model.feature_names_in_
        input_df = pd.DataFrame(
            [{k: feature_dict[k] for k in required_features}]
        )

    # Model probabilities
    probs = model.predict_proba(input_df)[0]

    # 1. Map team names to probabilities
    all_teams = team_encoder.classes_
    prob_dict = dict(zip(all_teams, probs))

    # 2. Get selected teams' probabilities
    p1 = float(prob_dict.get(team1, 0.0))
    p2 = float(prob_dict.get(team2, 0.0))

    # 3. Normalize to 100%
    total_prob = p1 + p2

    if total_prob > 0:
        t1_perc = round((p1 / total_prob) * 100, 2)
        t2_perc = round((p2 / total_prob) * 100, 2)
    else:
        t1_perc = 50.0
        t2_perc = 50.0

    # 4. Determine winner
    winner = team1 if p1 > p2 else team2
    confidence = max(t1_perc, t2_perc)

    # Display Probability Bar
    st.divider()
    st.subheader("🏆 Win Probability")

    components.html(
        f"""
        <style>
        .prob-card {{
            background-color: #1e1e24;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #444;
            font-family: sans-serif;
        }}

        .bar-labels {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-weight: 700;
            font-size: 14px;
        }}

        .bar-container {{
            width: 100%;
            height: 18px;
            background-color: #555;
            border-radius: 10px;
            display: flex;
            overflow: hidden;
        }}

        .bar-fill {{
            height: 100%;
            transition: width 1s ease-in-out;
        }}

        .bg-t1 {{ background-color: #4cc9f0; }}
        .bg-t2 {{ background-color: #f72585; }}

        .t1-color {{ color: #4cc9f0; }}
        .t2-color {{ color: #f72585; }}
        </style>

        <div class="prob-card">
            <div class="bar-labels">
                <span class="t1-color">{team1} ({t1_perc:.2f}%)</span>
                <span class="t2-color">{team2} ({t2_perc:.2f}%)</span>
            </div>

            <div class="bar-container">
                <div class="bar-fill bg-t1" style="width: {t1_perc}%;"></div>
                <div class="bar-fill bg-t2" style="width: {t2_perc}%;"></div>
            </div>
        </div>
        """,
        height=120
    )

    # Results
    if abs(t1_perc - t2_perc) < 5:
        st.info("⚖️ Very close match — could go either way")
    else:
        st.success(f"🏆 {winner} likely to win ({confidence:.2f}%)")