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
# HELPER FUNCTIONS
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


def auto_complete_first_innings(match_format):
    if match_format == "T20":
        return {
            "runs": random.randint(140, 200),
            "wkts": random.randint(4, 9),
            "overs": 20.0
        }

    if match_format == "ODI":
        return {
            "runs": random.randint(230, 330),
            "wkts": random.randint(5, 9),
            "overs": 50.0
        }

    # TEST: combined two innings
    return {
        "runs": random.randint(300, 550),
        "wkts": random.randint(10, 20),  # both innings combined
        "overs": None
    }

def realistic_projection(runs, overs, max_overs, wkts):
    if overs <= 0 or not max_overs:
        return runs

    current_rr = runs / overs

    # Format caps
    if max_overs == 20:
        rr_cap = 11.5
    elif max_overs == 50:
        rr_cap = 8.5
    else:
        rr_cap = current_rr

    wicket_factor = max(0.7, 1 - wkts * 0.05)
    effective_rr = min(current_rr, rr_cap) * wicket_factor

    return int(effective_rr * max_overs)

def pure_live_probability(curr_runs, curr_wkts, curr_overs, target, max_overs):
    if curr_overs <= 0 or target <= 0:
        return 0.5  # neutral before chase

    overs_left = max(max_overs - curr_overs, 0.1)
    runs_left = max(target - curr_runs, 0)

    current_rr = curr_runs / curr_overs
    required_rr = runs_left / overs_left

    # Core match pressure score
    rr_score = (current_rr - required_rr) / max(required_rr, 0.1)
    wicket_score = (10 - curr_wkts) / 10
    over_score = overs_left / max_overs

    # Heavy real-world weighting
    match_score = (
        rr_score * 0.60 +
        wicket_score * 0.25 +
        over_score * 0.15
    )

    # Convert to probability
    p = 0.5 + match_score * 0.45
    return min(max(p, 0.01), 0.99)

def display_rr(value, overs):
    if match_format == "TEST" or overs <= 0:
        return "—"
    return f"{value:.2f}"

def team_abbr(team_name: str) -> str:
    words = team_name.split()

    if len(words) > 1:
        return "".join(word[0] for word in words).upper()
    else:
        return team_name[:3].upper()

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
def init_state():

    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    if "auto_filled" not in st.session_state:
        st.session_state.auto_filled = False
    
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
        "inn1_runs": 0,
        "inn1_wkts": 0,
        "inn1_overs": 0.0,
        "inn2_runs": 0,
        "inn2_wkts": 0,
        "inn2_overs": 0.0,
    }
    
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
init_state()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    match_format = st.selectbox(
        "Match Format",
        ["T20", "ODI", "TEST"],
        index=1,
        placeholder="Select Format",
        key="match_format"
    )

    # Detect format change → hard reset
    if "prev_format" not in st.session_state:
        st.session_state.prev_format = match_format

    if st.session_state.prev_format != match_format:
        for k in list(st.session_state.keys()):
            del st.session_state[k]

        st.session_state.match_format = match_format
        st.session_state.prev_format = match_format
        st.session_state.team1_select = None
        st.session_state.team2_select = None
        st.session_state.submitted = False
        st.rerun()

    if st.button("🔄 Reset All", key="reset_btn"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.team1_select = None
        st.session_state.team2_select = None
        st.session_state.submitted = False
        st.rerun()

# --------------------------------------------------
# STEP 1: TEAMS
# --------------------------------------------------
st.divider()
st.subheader("1️⃣ Teams")
st.markdown("")

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
# STEP 2: VENUE & PITCH CONDITION
# --------------------------------------------------
st.divider()
st.subheader("2️⃣ Venue & Pitch")
st.markdown("")

c1, c2 = st.columns(2)

venue = c1.selectbox("Venue", venues, index=None, placeholder="Select Venue")
pitch_condition = c2.selectbox(
    "Pitch Condition",
    ["Balanced", "Batting Friendly", "Bowling Friendly"],
    index=None,
    placeholder="Select Pitch Condition"
)

st.session_state.venue = venue
st.session_state.pitch = pitch_condition

if (venue and pitch_condition):
    st.success(f"✅ Venue & Pitch Condition set!  \n\nVenue: **{venue}**  \nPitch: **{pitch_condition}**")
else:
    st.warning("👆 Please provide venue and pitch information")

if not (venue and pitch_condition):
    st.stop()

# --------------------------------------------------
# STEP 3: TOSS
# --------------------------------------------------
st.divider()
st.subheader("3️⃣ Toss")
st.markdown("")

c1, c2 = st.columns(2)

toss_winner = c1.selectbox("Toss Winner", [team1, team2], index=None, placeholder="Select Toss Winner")
toss_decision = c2.selectbox("Toss Decision", ["Batting", "Bowling"], index=None, placeholder="Select Toss Decision")

st.session_state.toss_winner = toss_winner
st.session_state.toss_decision = toss_decision

if (toss_winner and toss_decision):
    st.success(f"✅ Toss: **{toss_winner}** won the toss and chose to **{"Bat" if toss_decision == "Batting" else "Bowl"}** first")
else:    
    st.warning("👆 Please provide toss information")

if not (toss_winner and toss_decision):
    st.stop()

if not st.session_state.match_randomized:
    randomize_match()
    st.session_state.match_randomized = True

# --------------------------------------------------
# STEP 4: MATCH PROGRESS
# --------------------------------------------------
st.divider()
st.subheader("4️⃣ Match Details")
st.markdown("")

# AUTO-COMPLETE FIRST INNINGS
if not st.session_state.auto_filled:
    auto = auto_complete_first_innings(match_format)

    st.session_state.inn1_runs = auto["runs"]
    st.session_state.inn1_wkts = auto["wkts"]

    if match_format != "TEST":
        st.session_state.inn1_overs = auto["overs"]

    st.session_state.auto_filled = True

if toss_decision == "Batting":
    batting_first = toss_winner
    bowling_first = team2 if toss_winner == team1 else team1
else:
    bowling_first = toss_winner
    batting_first = team2 if toss_winner == team1 else team1

st.markdown(f"🏏 **Batting First:** {batting_first}")
st.markdown(f"🥎 **Bowling First:** {bowling_first}")
st.markdown("")

# Overs by format
if match_format == "T20":
    MAX_OVERS = 20
elif match_format == "ODI":
    MAX_OVERS = 50
else:
    MAX_OVERS = None  # TEST

col1, col2 = st.columns(2)

# SAFE DEFAULTS (important for TEST)
inn2_overs = 0.0
rr2 = 0.0
rrr = 0.0
runs_left = 0

# FIRST INNINGS
with col1:
    if match_format == "TEST":
        st.markdown(f"### Combined Innings – {team_abbr(batting_first)}")
    else:
        st.markdown(f"### First Innings – {team_abbr(batting_first)}")

    st.number_input("Total Runs", min_value=0, key="inn1_runs")
    max_wkts = 20 if match_format == "TEST" else 10
    st.number_input("Wickets Lost", min_value=0, max_value=max_wkts, key="inn1_wkts")

    inn1_runs = st.session_state.inn1_runs
    inn1_wkts = st.session_state.inn1_wkts

    if match_format != "TEST":
        st.number_input("Overs Played", min_value=0.1, max_value=float(MAX_OVERS), step=0.1, key="inn1_overs")

        inn1_overs = st.session_state.inn1_overs

        rr1 = inn1_runs / inn1_overs
        proj1 = realistic_projection(
            inn1_runs,
            inn1_overs,
            MAX_OVERS,
            inn1_wkts
        )

        t1_runs = proj1
        t1_wkts = min(10, inn1_wkts + 2)
    
    else:
        t1_runs = inn1_runs
        t1_wkts = inn1_wkts

# SECOND INNINGS
with col2:
    if match_format == "TEST":
        st.markdown(f"### Combined Innings – {team_abbr(bowling_first)}")
    else:
        st.markdown(f"### Second Innings – {team_abbr(bowling_first)}")

    target = inn1_runs + 1

    st.number_input("Current Runs", min_value=0, key="inn2_runs")
    max_wkts = 20 if match_format == "TEST" else 10
    st.number_input("Wickets Lost", min_value=0, max_value=max_wkts, key="inn2_wkts")

    inn2_runs = st.session_state.inn2_runs
    inn2_wkts = st.session_state.inn2_wkts

    if match_format != "TEST":
        st.number_input("Overs Played", min_value=0.1, max_value=float(MAX_OVERS), step=0.1, key="inn2_overs")

        inn2_overs = st.session_state.inn2_overs

        overs_left = max(MAX_OVERS - inn2_overs, 0.1)
        runs_left = max(target - inn2_runs, 0)

        rr2 = inn2_runs / max(inn2_overs, 0.1)
        rrr = runs_left / overs_left

        proj2 = realistic_projection(
            inn2_runs,
            max(inn2_overs, 0.1),
            MAX_OVERS,
            inn2_wkts
        )

        t2_runs = proj2
        t2_wkts = inn2_wkts
    
    else:
        t2_runs = inn2_runs
        t2_wkts = inn2_wkts

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.markdown("")
if st.button("🚀 Submit"):
    st.session_state.submitted = True

    # Encoding
    t1_enc = team_encoder.transform([team1])[0]
    t2_enc = team_encoder.transform([team2])[0]
    venue_enc = venue_encoder.transform([venue])[0]

    toss_enc = team_encoder.transform([toss_winner])[0]
    toss_dec_enc = 1 if toss_decision == "Batting" else 0

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

    # Map team names to probabilities
    all_teams = team_encoder.classes_
    prob_dict = dict(zip(all_teams, probs))

    # Get selected teams' probabilities
    p1 = float(prob_dict.get(team1, 0.0))
    p2 = float(prob_dict.get(team2, 0.0))

    # Identify probabilities correctly
    if batting_first == team1:
        p_batting = p1
        p_chasing = p2
    else:
        p_batting = p2
        p_chasing = p1

    if match_format in ["T20", "ODI"]:
        p_chasing = pure_live_probability(
            curr_runs=inn2_runs,
            curr_wkts=inn2_wkts,
            curr_overs=inn2_overs,
            target=target,
            max_overs=MAX_OVERS
        )
        p_batting = 1 - p_chasing
    else:
        p_batting, p_chasing = p_batting, p_chasing


    # Final probability mapping
    if batting_first == team1:
        p1 = p_batting
        p2 = p_chasing
    else:
        p2 = p_batting
        p1 = p_chasing

    t1_perc = round(p1 * 100, 2)
    t2_perc = round(p2 * 100, 2)


    # Determine winner
    winner = team1 if p1 > p2 else team2
    confidence = max(t1_perc, t2_perc)

    # MATCH SUMMARY
    if st.session_state.submitted:

        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        st.markdown("")

        # Safe RR values
        display_current_rr = (
            rr2 if inn2_overs > 0 else 0.0
        )

        display_required_rr = (
            rrr if match_format != "TEST" else 0.0
        )

        display_target = target if match_format != "TEST" else "—"
        display_runs_needed = runs_left if match_format != "TEST" else "—"

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Current RR", display_rr(rr2, inn2_overs))
        c2.metric("Required RR", display_rr(rrr, inn2_overs))
        c3.metric("Target", display_target)
        c4.metric("Runs Needed", display_runs_needed)

        st.markdown("")

        p1_col, p2_col = st.columns(2)

        p1_col.metric(
            f"{team1} Win Probability",
            f"{t1_perc}%"
        )

        p2_col.metric(
            f"{team2} Win Probability",
            f"{t2_perc}%"
        )

    # Display Probability Bar
    st.markdown("")
    st.markdown("")
    st.subheader("🏆 Probability Bar")
    st.markdown("")

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
