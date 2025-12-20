from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import webbrowser
from threading import Timer
from datetime import datetime

app = Flask(__name__)

# --- 1. Load the Saved Model ---
try:
    with open('cricket_prediction_model.pkl', 'rb') as f:
        data = pickle.load(f)
    
    model_odi = data["model_odi"]
    model_t20 = data["model_t20"]
    model_test = data["model_test"]
    le = data["encoder"]
    
    # Teams list for Dropdown
    teams = sorted(le.classes_)
    
except Exception as e:
    print(f"Error loading model: {e}")
    teams = []

# --- 2. Routes ---

@app.route('/')
def home():
    # Shuru mein context empty hoga
    return render_template('index.html', teams=teams, context={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- 1. Form Data Capture (No Changes) ---
        match_format = request.form.get('match_format')
        team1 = request.form.get('team1')
        team2 = request.form.get('team2')
        
        # --- Validation (No Changes) ---
        if not match_format or not team1 or not team2:
             return render_template('index.html', prediction_text="⚠️ Please select Match Format and Both Teams.", teams=teams, context=request.form)
        
        if team1 == team2:
            return render_template('index.html', prediction_text="⚠️ Please select two different teams.", teams=teams, context=request.form)

        # --- Advanced Inputs (No Changes) ---
        try:
            venue_score = float(request.form.get('venue_score', 0))
            toss_winner = request.form.get('toss_winner', team1)
            toss_decision = request.form.get('toss_decision', 'Batting')
            pitch_condition = request.form.get('pitch_condition', 'Balanced')
            
            t1_score = float(request.form.get('t1_score', 0))
            t1_wkt = float(request.form.get('t1_wkt', 0))
            t2_score = float(request.form.get('t2_score', 0))
            t2_wkt = float(request.form.get('t2_wkt', 0))
        except ValueError:
             return render_template('index.html', prediction_text="⚠️ Error: Invalid input values.", teams=teams, context=request.form)

        # --- Calculations & Encoding (No Changes) ---
        
        # Toss Fallback
        if toss_winner not in le.classes_: toss_winner = team1
        
        t1_enc = le.transform([team1])[0]
        t2_enc = le.transform([team2])[0]
        toss_win_enc = le.transform([toss_winner])[0]
        toss_dec_enc = 1 if toss_decision == "Batting" else 0
        
        # Run Rates
        t1_rr = t1_score / t1_wkt if t1_wkt > 0 else t1_score
        t2_rr = t2_score / t2_wkt if t2_wkt > 0 else t2_score
        
        # Pitch Encoding
        pitch_bat = 1 if pitch_condition == "Batting Friendly" else 0
        pitch_bowl = 1 if pitch_condition == "Bowling Friendly" else 0
        
        # Date Logic
        current_month = datetime.now().month

        # ---------------------------------------------------------
        # --- CHANGED: Prediction Logic (Probability Calculation) ---
        # ---------------------------------------------------------
        
        probs = [] # Isme hum probabilities store karenge

        if match_format == "Test":
            # Test Model Columns
            feature_columns = [
                'Team1 Run Rate', 'Team2 Run Rate', 'Match Start Month', 
                'Match End Month', 'Diff Month', 'Toss Decision Encoded', 
                'Pitch_Batting Friendly', 'Pitch_Bowling Friendly', 
                'Team1 Encoded', 'Team2 Encoded', 'Toss Winner Encoded'
            ]
            
            input_data = pd.DataFrame([[
                t1_rr, t2_rr, current_month, current_month, 0, 
                toss_dec_enc, bool(pitch_bat), bool(pitch_bowl), 
                t1_enc, t2_enc, toss_win_enc
            ]], columns=feature_columns)
            
            # CHANGE: predict_proba use kiya
            probs = model_test.predict_proba(input_data)[0]

        else:
            # T20/ODI Model Columns
            feature_columns = [
                'Venue Average Score', 'Team1 Run Rate', 'Team2 Run Rate', 
                'Match Month', 'Toss Decision Encoded', 'Pitch_Batting Friendly', 
                'Pitch_Bowling Friendly', 'Team1 Encoded', 'Team2 Encoded', 
                'Toss Winner Encoded'
            ]
            
            input_data = pd.DataFrame([[
                venue_score, t1_rr, t2_rr, current_month, 
                toss_dec_enc, bool(pitch_bat), bool(pitch_bowl), 
                t1_enc, t2_enc, toss_win_enc
            ]], columns=feature_columns)
            
            # CHANGE: predict_proba use kiya based on format
            if match_format == "T20":
                probs = model_t20.predict_proba(input_data)[0]
            else:
                probs = model_odi.predict_proba(input_data)[0]

        # ---------------------------------------------------------
        # --- NEW: Calculating Percentage for Selected Teams ---
        # ---------------------------------------------------------
        
        # 1. Team Names aur unki Probabilities ko map kiya
        all_teams = le.classes_
        prob_dict = dict(zip(all_teams, probs))

        # 2. Selected Teams ki probability nikali
        p1 = prob_dict.get(team1, 0.0)
        p2 = prob_dict.get(team2, 0.0)

        # 3. Google Style Percentage (Normalize to 100%)
        # Agar P1=0.4 aur P2=0.4 hai (matlab baaki 0.2 draw/other hai),
        # toh hum bar chart ke liye in dono ko 100% scale par le aate hain.
        total_prob = p1 + p2
        
        if total_prob > 0:
            t1_perc = round((p1 / total_prob) * 100, 2)
            t2_perc = round((p2 / total_prob) * 100, 2)
        else:
            # Fallback agar kuch garbar ho (e.g. Draw chances very high)
            t1_perc = 50
            t2_perc = 50

        # 4. Winner Determine karna (based on higher probability)
        if p1 > p2:
            winner = team1
        else:
            winner = team2

        # ---------------------------------------------------------
        # --- Return Result with Percentages ---
        # ---------------------------------------------------------
        return render_template('index.html', 
                               prediction_text=f"🏆 {winner} is likely to win", 
                               team1=team1, 
                               team2=team2, 
                               t1_perc=t1_perc, 
                               t2_perc=t2_perc,
                               teams=teams, 
                               context=request.form)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}", teams=teams, context=request.form)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)