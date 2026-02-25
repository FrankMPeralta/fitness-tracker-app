# Personal Fitness Tracker

Beginner-friendly local app to track:

- Food by grams
- Running totals for calories, protein, carbs, and fat
- Weight, body fat %, and steps

The app runs locally and stores data in SQLite (`fitness_tracker.db`).

## Setup

```bash
cd "/Users/frankmperalta/Documents/New project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Main Pages

- `Food Log`: choose food, enter grams or quantity (for count-based foods like eggs/scoops), app calculates macros and keeps running daily totals.
- `Manage Foods`: add custom foods with serving-size macros and choose input mode (`grams` or `quantity`).
- `Body Metrics`: log weight/body-fat/steps.
- `Dashboard`: trend charts for body metrics and daily macro totals.
- `Export`: export body metrics and daily macro totals as CSV.

## Food Library

The app is pre-seeded with starter foods from your current tracker table and allows custom foods.

For each food, you can store:

- Name
- Category
- Grams per serving
- Calories per serving
- Protein per serving
- Carbs per serving
- Fat per serving

## Notes

- If you log the same food multiple times in a day, totals accumulate.
- You can delete incorrect food log entries from the `Food Log` page.
- CSV import currently targets body metrics only.
