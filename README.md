# Fantasy IQ

Fantasy IQ is a real-time Fantasy Premier League decision-support application built with **Streamlit**, **machine learning**, and **optimization**.

The app is designed to help FPL managers make smarter decisions for the current gameweek by combining live official FPL data with predictive modeling and rule-based squad optimization.

---

## Features

Fantasy IQ consists of 3 main modules:

### 1. Prediction Engine
The Prediction Engine allows users to:
- view predicted next-gameweek points for current Premier League players
- filter players by position, club, and price
- compare players using live model outputs

### 2. Squad Builder
The Squad Builder contains 2 tabs:

#### Best Current Squad
- builds the best possible 15-player squad based on current live predicted points
- follows official FPL squad rules
- displays the best valid starting XI and bench on a football pitch

#### GW1 Squad Builder
- helps managers build their initial squad for the beginning of a season
- uses archived previous-season player performance with a hybrid scoring approach
- supports unmatched players through fallback scoring
- generates a valid opening 15-player squad under FPL constraints

### 3. Transfer Assistant
The Transfer Assistant allows users to:
- input their current 15-player squad by position
- validate the squad
- choose a starting XI
- view current squad and starting XI predicted points
- generate the best 1-transfer and 2-transfer recommendations

---

## How It Works

The project follows this pipeline:

1. **Live Data Collection**  
   Retrieves player, team, fixture, and player-history data from the official FPL API.

2. **Data Preprocessing**  
   Cleans and structures the raw API data into usable historical tables.

3. **Feature Engineering**  
   Rebuilds lag, rolling, consistency, and availability features from live player history.

4. **Model Inference**  
   Loads the final trained models and generates predicted next-gameweek points.

5. **Optimization**  
   Uses FPL constraints to build the best squads and evaluate transfer options.

6. **Interactive Application**  
   Presents everything through a Streamlit app with filters, tables, metrics, and pitch visualizations.

---

## Final Models Used

The final position-based models used in deployment are:

- **Goalkeepers:** Linear Regression
- **Defenders:** Linear Regression
- **Midfielders:** Linear Regression
- **Forwards:** LSTM

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- PuLP
- Requests

---

## Project Structure

```bash
fantasy-iq/
│
├── app.py
├── requirements.txt
├── models/
├── src/
├── data_cache/
└── notebooks/
```
---   
## Running the App Locally
1) Clone the repository
2) Install the required packages
```bash
pip install -r requirements.txt
```
3) Run the app
```bash
streamlit run app.py
```
---

## Deployment
The project was deployed using:
- Github for version control and hosting
- Streamlit Community Cloud for web deployment

---

## Notes
1) The app uses live FPL Data, so the first run may take a few minutes
2) After the first run, caching makes repeated interactions faster
3) In the Transfer Assistant, the 100 Million starting budget is not enforced, becasue player prices change throughout the season

---

## Purpose
Fantasy IQ was developed as a graduation project to demonstrate how ML  & Optimization can be combined into real-time sports analytics decision support systems

The Goal of this project is not only to predict player performance, but also to turn those predictions into practical fantasy football decisions through an interactive deployed application
