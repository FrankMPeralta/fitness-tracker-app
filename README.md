# Personal Fitness Tracker (Supabase)

This app now uses Supabase (Postgres) so your data persists and syncs across devices (laptop + iPhone web app).

## 1) Supabase setup

Create a Supabase project, then run this SQL in Supabase SQL Editor:

```sql
create table if not exists daily_entries (
  entry_date date primary key,
  weight_lbs numeric,
  body_fat_pct numeric,
  steps integer,
  calories integer,
  food_notes text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists foods (
  id bigint generated always as identity primary key,
  name text unique not null,
  category text not null,
  grams_per_serving numeric not null,
  cals_per_serving numeric not null,
  protein_per_serving numeric not null,
  carbs_per_serving numeric not null,
  fat_per_serving numeric not null,
  input_mode text not null default 'grams',
  unit_label text not null default 'serving',
  units_per_serving numeric not null default 1,
  created_at timestamptz default now()
);

create table if not exists food_logs (
  id bigint generated always as identity primary key,
  entry_date date not null,
  food_id bigint not null references foods(id) on delete cascade,
  grams numeric not null,
  servings numeric not null,
  cals numeric not null,
  protein numeric not null,
  carbs numeric not null,
  fat numeric not null,
  notes text,
  created_at timestamptz default now()
);

create table if not exists macro_goals (
  id integer primary key,
  goal_name text not null,
  target_cals numeric not null,
  target_protein numeric not null,
  target_carbs numeric not null,
  target_fat numeric not null,
  body_weight numeric,
  body_fat_pct numeric,
  bmr numeric,
  daily_activity numeric,
  updated_at timestamptz default now()
);
```

## 2) Add local secrets

Create `.streamlit/secrets.toml` in this project:

```toml
SUPABASE_URL = "https://YOUR_PROJECT_ID.supabase.co"
SUPABASE_ANON_KEY = "YOUR_SUPABASE_ANON_KEY"
```

## 3) Install + run

```bash
cd "/Users/frankmperalta/Desktop/New project"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## 4) Migrate your existing local SQLite data (optional)

If you already have data in `fitness_tracker.db`:

```bash
python3 migrate_sqlite_to_supabase.py
```

This will copy:
- foods
- daily_entries
- food_logs

## Main Pages

- `Food Log`: log foods by grams or quantity (eggs, scoops, slices).
- `Manage Foods`: add custom foods and choose input mode.
- `Body Metrics`: weight, body-fat, steps.
- `Macro Calculator`: maintenance/deficit planning.
- `Dashboard`: trends and daily macro totals.
- `Import CSV`: import body metrics.
- `Export`: export body metrics + daily macro totals.


## Goal Tracking

Macro goals are now saved from `Macro Calculator` into `macro_goals` and used in `Dashboard` and `Food Log` for goal-vs-actual progress.
