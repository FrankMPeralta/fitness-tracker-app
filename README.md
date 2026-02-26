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

## Auth + RLS Migration (Security)

Run this in Supabase SQL Editor to enable per-user data isolation with Supabase Auth.

```sql
alter table daily_entries add column if not exists user_id uuid references auth.users(id) on delete cascade;
alter table foods add column if not exists user_id uuid references auth.users(id) on delete cascade;
alter table food_logs add column if not exists user_id uuid references auth.users(id) on delete cascade;
alter table macro_goals add column if not exists user_id uuid references auth.users(id) on delete cascade;

-- One-time legacy backfill for a single-user app.
-- If you have multiple users already, set user_id per user instead of this blanket update.
update daily_entries set user_id = (select id from auth.users order by created_at limit 1) where user_id is null;
update foods set user_id = (select id from auth.users order by created_at limit 1) where user_id is null;
update food_logs set user_id = (select id from auth.users order by created_at limit 1) where user_id is null;
update macro_goals set user_id = (select id from auth.users order by created_at limit 1) where user_id is null;

alter table foods drop constraint if exists foods_name_key;

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'foods_user_id_name_key'
  ) then
    alter table foods add constraint foods_user_id_name_key unique (user_id, name);
  end if;
end $$;

alter table daily_entries drop constraint if exists daily_entries_pkey;

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'daily_entries_user_id_entry_date_key'
  ) then
    alter table daily_entries add constraint daily_entries_user_id_entry_date_key unique (user_id, entry_date);
  end if;
end $$;

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'macro_goals_user_id_key'
  ) then
    alter table macro_goals add constraint macro_goals_user_id_key unique (user_id);
  end if;
end $$;

alter table daily_entries enable row level security;
alter table foods enable row level security;
alter table food_logs enable row level security;
alter table macro_goals enable row level security;

drop policy if exists daily_entries_isolation on daily_entries;
create policy daily_entries_isolation on daily_entries
for all using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists foods_isolation on foods;
create policy foods_isolation on foods
for all using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists food_logs_isolation on food_logs;
create policy food_logs_isolation on food_logs
for all using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists macro_goals_isolation on macro_goals;
create policy macro_goals_isolation on macro_goals
for all using (auth.uid() = user_id)
with check (auth.uid() = user_id);
```

After running the SQL:
1. Reboot Streamlit Cloud app.
2. Sign in / create account in the app.
3. If your data was already in the tables, verify those rows now have user_id set after running the migration SQL.
