from __future__ import annotations

import os
from datetime import date, datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st
from supabase import Client, create_client

ALIASES = {
    "date": ["date", "day"],
    "weight_lbs": ["weight_lbs", "weight", "body_weight", "body weight", "lbs"],
    "body_fat_pct": ["body_fat_pct", "body fat", "body_fat", "bf%", "body fat %"],
    "steps": ["steps", "step_count", "step count"],
    "calories": ["calories", "kcal", "cals", "total_calories", "total calories"],
    "food_notes": ["food_notes", "food", "notes", "food note", "meal_notes", "meals"],
}

DEFAULT_FOODS = [
    {"name": "Eggs", "category": "Protein", "grams_per_serving": 50, "cals": 70, "protein": 6, "carbs": 1, "fat": 5},
    {"name": "Egg Whites", "category": "Protein", "grams_per_serving": 46, "cals": 25, "protein": 5, "carbs": 0, "fat": 0},
    {"name": "Protein Shake (1 Scoop)", "category": "Protein", "grams_per_serving": 1, "cals": 120, "protein": 25, "carbs": 2, "fat": 1.5},
    {"name": "Elk Steak", "category": "Protein", "grams_per_serving": 113, "cals": 130, "protein": 26, "carbs": 0, "fat": 1.5},
    {"name": "Ground Elk", "category": "Protein", "grams_per_serving": 100, "cals": 160, "protein": 25, "carbs": 0, "fat": 6},
    {"name": "Chicken Breast", "category": "Protein", "grams_per_serving": 100, "cals": 165, "protein": 31, "carbs": 0, "fat": 3.5},
    {"name": "Chicken Thigh", "category": "Protein", "grams_per_serving": 100, "cals": 177, "protein": 24, "carbs": 0, "fat": 8},
    {"name": "Tilapia", "category": "Protein", "grams_per_serving": 100, "cals": 128, "protein": 26, "carbs": 0, "fat": 3},
    {"name": "Salmon", "category": "Protein", "grams_per_serving": 100, "cals": 208, "protein": 20, "carbs": 0, "fat": 13},
    {"name": "Greek Yogurt (Oikos Triple Zero)", "category": "Protein", "grams_per_serving": 170, "cals": 90, "protein": 15, "carbs": 7, "fat": 0},
    {"name": "Ground Turkey", "category": "Protein", "grams_per_serving": 100, "cals": 203, "protein": 27, "carbs": 0, "fat": 10},
    {"name": "Shrimp", "category": "Protein", "grams_per_serving": 113, "cals": 80, "protein": 18, "carbs": 0, "fat": 0},
    {"name": "Cottage Cheese", "category": "Protein", "grams_per_serving": 110, "cals": 110, "protein": 15, "carbs": 3, "fat": 7},
    {"name": "LMNT Electrolyte", "category": "Carbs", "grams_per_serving": 1, "cals": 10, "protein": 0, "carbs": 2, "fat": 0},
    {"name": "Quick Oats (40g uncooked)", "category": "Carbs", "grams_per_serving": 40, "cals": 150, "protein": 5, "carbs": 27, "fat": 2.5},
    {"name": "Minute Rice", "category": "Carbs", "grams_per_serving": 85, "cals": 170, "protein": 4, "carbs": 38, "fat": 0},
    {"name": "Sweet Potato", "category": "Carbs", "grams_per_serving": 100, "cals": 90, "protein": 2, "carbs": 21, "fat": 0},
    {"name": "Dave's Bread (1 slice)", "category": "Carbs", "grams_per_serving": 55, "cals": 110, "protein": 6, "carbs": 22, "fat": 1.5},
    {"name": "Greek Yogurt", "category": "Carbs", "grams_per_serving": 170, "cals": 100, "protein": 18, "carbs": 6, "fat": 0},
    {"name": "Pineapple", "category": "Carbs", "grams_per_serving": 100, "cals": 52, "protein": 0.3, "carbs": 14, "fat": 0},
    {"name": "Blueberries", "category": "Carbs", "grams_per_serving": 100, "cals": 57, "protein": 0.7, "carbs": 14.5, "fat": 0},
    {"name": "Banana", "category": "Carbs", "grams_per_serving": 100, "cals": 89, "protein": 1.1, "carbs": 23, "fat": 0},
    {"name": "Broccoli", "category": "Carbs", "grams_per_serving": 85, "cals": 30, "protein": 3, "carbs": 5, "fat": 0},
    {"name": "Strawberry", "category": "Carbs", "grams_per_serving": 140, "cals": 50, "protein": 1, "carbs": 10, "fat": 0},
    {"name": "Sweet Pepper", "category": "Carbs", "grams_per_serving": 85, "cals": 30, "protein": 1, "carbs": 6, "fat": 0},
    {"name": "Carrot", "category": "Carbs", "grams_per_serving": 100, "cals": 41, "protein": 1, "carbs": 10, "fat": 0},
    {"name": "Pasta Sauce", "category": "Carbs", "grams_per_serving": 118, "cals": 60, "protein": 2, "carbs": 7, "fat": 2.5},
    {"name": "Almonds", "category": "Fats", "grams_per_serving": 100, "cals": 579, "protein": 21, "carbs": 22, "fat": 50},
    {"name": "Avocado", "category": "Fats", "grams_per_serving": 100, "cals": 320, "protein": 3, "carbs": 12, "fat": 21},
    {"name": "Butter", "category": "Fats", "grams_per_serving": 14, "cals": 100, "protein": 0, "carbs": 0, "fat": 11},
    {"name": "Sour Cream", "category": "Fats", "grams_per_serving": 30, "cals": 60, "protein": 1, "carbs": 2, "fat": 5},
    {"name": "Pitted Olives", "category": "Fats", "grams_per_serving": 15, "cals": 30, "protein": 1, "carbs": 2, "fat": 3},
    {"name": "Peanut Butter", "category": "Fats", "grams_per_serving": 32, "cals": 180, "protein": 8, "carbs": 5, "fat": 16},
    {"name": "Frozen Veggies", "category": "Carbs", "grams_per_serving": 85, "cals": 60, "protein": 3, "carbs": 11, "fat": 0.5},
]


@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets.get("SUPABASE_URL") if "SUPABASE_URL" in st.secrets else os.getenv("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_ANON_KEY") if "SUPABASE_ANON_KEY" in st.secrets else os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_ANON_KEY")
    return create_client(url, key)


def require_supabase() -> Client | None:
    try:
        return get_supabase_client()
    except Exception:
        st.error("Supabase is not configured.")
        st.code(
            """In Streamlit secrets (or local env), set:\nSUPABASE_URL=...\nSUPABASE_ANON_KEY=..."""
        )
        return None


def default_food_input_config(food_name: str) -> tuple[str, str, float]:
    name = food_name.strip().lower()
    if name == "eggs":
        return ("quantity", "egg", 1.0)
    if name == "egg whites":
        return ("quantity", "serving", 1.0)
    if "protein shake" in name:
        return ("quantity", "scoop", 1.0)
    if name == "lmnt electrolyte":
        return ("quantity", "packet", 1.0)
    if "bread" in name and "slice" in name:
        return ("quantity", "slice", 1.0)
    return ("grams", "serving", 1.0)


def seed_default_foods(client: Client) -> None:
    rows = []
    for food in DEFAULT_FOODS:
        mode, unit_label, units_per_serving = default_food_input_config(food["name"])
        rows.append(
            {
                "name": food["name"],
                "category": food["category"],
                "grams_per_serving": food["grams_per_serving"],
                "cals_per_serving": food["cals"],
                "protein_per_serving": food["protein"],
                "carbs_per_serving": food["carbs"],
                "fat_per_serving": food["fat"],
                "input_mode": mode,
                "unit_label": unit_label,
                "units_per_serving": units_per_serving,
            }
        )

    client.table("foods").upsert(rows, on_conflict="name").execute()


def normalize_import_columns(df: pd.DataFrame) -> pd.DataFrame:
    original = list(df.columns)
    normalized = {c: str(c).strip().lower() for c in original}
    rename_map = {}

    for target, options in ALIASES.items():
        option_set = {o.lower() for o in options}
        for col, low in normalized.items():
            if low in option_set:
                rename_map[col] = target
                break

    out = df.rename(columns=rename_map).copy()

    for col in ["date", "weight_lbs", "body_fat_pct", "steps", "calories", "food_notes"]:
        if col not in out.columns:
            out[col] = None

    keep = ["date", "weight_lbs", "body_fat_pct", "steps", "calories", "food_notes"]
    out = out[keep]

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out[out["date"].notna()].copy()

    out["weight_lbs"] = pd.to_numeric(out["weight_lbs"], errors="coerce")
    out["body_fat_pct"] = pd.to_numeric(out["body_fat_pct"], errors="coerce")
    out["steps"] = pd.to_numeric(out["steps"], errors="coerce").round()
    out["calories"] = pd.to_numeric(out["calories"], errors="coerce").round()
    out["food_notes"] = out["food_notes"].fillna("").astype(str)

    return out


def upsert_daily_entry(
    client: Client,
    entry_date: date,
    weight_lbs: float | None,
    body_fat_pct: float | None,
    steps: int | None,
    calories: int | None,
    food_notes: str,
) -> None:
    payload = {
        "entry_date": entry_date.isoformat(),
        "weight_lbs": weight_lbs,
        "body_fat_pct": body_fat_pct,
        "steps": steps,
        "calories": calories,
        "food_notes": food_notes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    client.table("daily_entries").upsert(payload, on_conflict="entry_date").execute()


def fetch_daily_entries(client: Client) -> pd.DataFrame:
    data = (
        client.table("daily_entries")
        .select("entry_date, weight_lbs, body_fat_pct, steps, calories, food_notes")
        .order("entry_date")
        .execute()
        .data
        or []
    )
    if not data:
        return pd.DataFrame(columns=["date", "weight_lbs", "body_fat_pct", "steps", "calories", "food_notes"])

    df = pd.DataFrame(data).rename(columns={"entry_date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_foods(client: Client) -> pd.DataFrame:
    data = (
        client.table("foods")
        .select(
            "id, name, category, grams_per_serving, cals_per_serving, protein_per_serving, "
            "carbs_per_serving, fat_per_serving, input_mode, unit_label, units_per_serving"
        )
        .order("category")
        .order("name")
        .execute()
        .data
        or []
    )

    cols = [
        "id",
        "name",
        "category",
        "grams_per_serving",
        "cals_per_serving",
        "protein_per_serving",
        "carbs_per_serving",
        "fat_per_serving",
        "input_mode",
        "unit_label",
        "units_per_serving",
    ]
    if not data:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(data)


def log_food_entry(client: Client, entry_date: date, food_id: int, grams: float, notes: str) -> None:
    rows = (
        client.table("foods")
        .select("grams_per_serving, cals_per_serving, protein_per_serving, carbs_per_serving, fat_per_serving")
        .eq("id", food_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        raise ValueError("Food not found")

    food = rows[0]
    grams_per_serving = float(food["grams_per_serving"])
    servings = grams / grams_per_serving if grams_per_serving > 0 else 0

    payload = {
        "entry_date": entry_date.isoformat(),
        "food_id": int(food_id),
        "grams": float(grams),
        "servings": float(servings),
        "cals": float(servings * float(food["cals_per_serving"])),
        "protein": float(servings * float(food["protein_per_serving"])),
        "carbs": float(servings * float(food["carbs_per_serving"])),
        "fat": float(servings * float(food["fat_per_serving"])),
        "notes": notes,
    }

    client.table("food_logs").insert(payload).execute()


def fetch_food_logs(client: Client, entry_date: date) -> pd.DataFrame:
    logs = (
        client.table("food_logs")
        .select("id, food_id, grams, servings, cals, protein, carbs, fat, notes")
        .eq("entry_date", entry_date.isoformat())
        .order("id", desc=True)
        .execute()
        .data
        or []
    )
    if not logs:
        return pd.DataFrame(columns=["id", "name", "category", "grams", "servings", "cals", "protein", "carbs", "fat", "notes"])

    logs_df = pd.DataFrame(logs)
    food_ids = logs_df["food_id"].dropna().astype(int).unique().tolist()

    foods = (
        client.table("foods")
        .select("id, name, category")
        .in_("id", food_ids)
        .execute()
        .data
        or []
    )
    foods_df = pd.DataFrame(foods)

    merged = logs_df.merge(foods_df, left_on="food_id", right_on="id", how="left", suffixes=("", "_food"))
    merged = merged[["id", "name", "category", "grams", "servings", "cals", "protein", "carbs", "fat", "notes"]]
    return merged


def delete_food_log(client: Client, log_id: int) -> None:
    client.table("food_logs").delete().eq("id", log_id).execute()


def fetch_daily_macro_totals(client: Client) -> pd.DataFrame:
    logs = (
        client.table("food_logs")
        .select("entry_date, cals, protein, carbs, fat")
        .order("entry_date")
        .execute()
        .data
        or []
    )

    if not logs:
        return pd.DataFrame(columns=["date", "cals", "protein", "carbs", "fat"])

    df = pd.DataFrame(logs).rename(columns={"entry_date": "date"})
    for col in ["cals", "protein", "carbs", "fat"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    grouped = (
        df.groupby("date", as_index=False)[["cals", "protein", "carbs", "fat"]]
        .sum()
        .sort_values("date")
    )
    grouped["date"] = pd.to_datetime(grouped["date"])
    return grouped


def add_food(
    client: Client,
    name: str,
    category: str,
    grams_per_serving: float,
    cals_per_serving: float,
    protein_per_serving: float,
    carbs_per_serving: float,
    fat_per_serving: float,
    input_mode: str,
    unit_label: str,
    units_per_serving: float,
) -> None:
    client.table("foods").insert(
        {
            "name": name.strip(),
            "category": category,
            "grams_per_serving": grams_per_serving,
            "cals_per_serving": cals_per_serving,
            "protein_per_serving": protein_per_serving,
            "carbs_per_serving": carbs_per_serving,
            "fat_per_serving": fat_per_serving,
            "input_mode": input_mode,
            "unit_label": unit_label.strip() or "serving",
            "units_per_serving": units_per_serving,
        }
    ).execute()


def import_rows(client: Client, df: pd.DataFrame) -> int:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "entry_date": row["date"].isoformat(),
                "weight_lbs": None if pd.isna(row["weight_lbs"]) else float(row["weight_lbs"]),
                "body_fat_pct": None if pd.isna(row["body_fat_pct"]) else float(row["body_fat_pct"]),
                "steps": None if pd.isna(row["steps"]) else int(row["steps"]),
                "calories": None if pd.isna(row["calories"]) else int(row["calories"]),
                "food_notes": row["food_notes"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    if rows:
        client.table("daily_entries").upsert(rows, on_conflict="entry_date").execute()
    return len(rows)


def latest_metric(df: pd.DataFrame, col: str):
    if df.empty:
        return None
    non_null = df.dropna(subset=[col])
    if non_null.empty:
        return None
    return non_null.iloc[-1][col]


def parse_optional_float(value: str) -> float | None:
    v = value.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_optional_int(value: str) -> int | None:
    v = value.strip()
    if not v:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def macro_breakdown(total_cals: float, body_weight: float, protein_ratio: float, fat_ratio: float) -> dict:
    protein_g = body_weight * protein_ratio
    fat_g = body_weight * fat_ratio
    protein_cals = protein_g * 4
    fat_cals = fat_g * 9
    remaining_cals = total_cals - protein_cals - fat_cals
    carb_cals = max(0.0, remaining_cals)
    carbs_g = carb_cals / 4
    return {
        "protein_g": protein_g,
        "fat_g": fat_g,
        "protein_cals": protein_cals,
        "fat_cals": fat_cals,
        "pf_cals": protein_cals + fat_cals,
        "carb_cals": carb_cals,
        "carbs_g": carbs_g,
        "remaining_cals": remaining_cals,
    }


def fetch_active_macro_goal(client: Client) -> dict | None:
    try:
        rows = (
            client.table("macro_goals")
            .select(
                "id, goal_name, target_cals, target_protein, target_carbs, target_fat, "
                "body_weight, body_fat_pct, bmr, daily_activity, updated_at"
            )
            .eq("id", 1)
            .limit(1)
            .execute()
            .data
            or []
        )
    except Exception:
        return None

    return rows[0] if rows else None


def save_active_macro_goal(
    client: Client,
    goal_name: str,
    target_cals: float,
    target_protein: float,
    target_carbs: float,
    target_fat: float,
    body_weight: float,
    body_fat_pct: float,
    bmr: float,
    daily_activity: float,
) -> None:
    payload = {
        "id": 1,
        "goal_name": goal_name,
        "target_cals": target_cals,
        "target_protein": target_protein,
        "target_carbs": target_carbs,
        "target_fat": target_fat,
        "body_weight": body_weight,
        "body_fat_pct": body_fat_pct,
        "bmr": bmr,
        "daily_activity": daily_activity,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    client.table("macro_goals").upsert(payload, on_conflict="id").execute()


def totals_for_date(macro_df: pd.DataFrame, target_date: date) -> dict:
    if macro_df.empty:
        return {"cals": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}

    day_rows = macro_df[macro_df["date"].dt.date == target_date]
    if day_rows.empty:
        return {"cals": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}

    row = day_rows.iloc[-1]
    return {
        "cals": float(row["cals"]),
        "protein": float(row["protein"]),
        "carbs": float(row["carbs"]),
        "fat": float(row["fat"]),
    }


def render_goals_vs_actual(actual: dict, goals: dict, section_title: str) -> None:
    st.markdown(f"### {section_title}")

    metrics = [
        ("Calories", "cals", "target_cals", "kcal"),
        ("Protein", "protein", "target_protein", "g"),
        ("Carbs", "carbs", "target_carbs", "g"),
        ("Fat", "fat", "target_fat", "g"),
    ]

    cols = st.columns(4)
    for col, (label, a_key, g_key, unit) in zip(cols, metrics):
        actual_val = float(actual.get(a_key, 0.0))
        goal_val = float(goals.get(g_key, 0.0) or 0.0)
        pct = (actual_val / goal_val * 100.0) if goal_val > 0 else 0.0

        with col:
            if label == "Calories" and goal_val > 0:
                delta = goal_val - actual_val
                delta_text = f"{delta:.0f} remaining"
                st.metric(label, f"{actual_val:.0f} / {goal_val:.0f}", delta=delta_text)
                color = "#16a34a" if actual_val <= goal_val else "#dc2626"
                status = "Under/On Target" if actual_val <= goal_val else "Over Target"
                st.markdown(
                    f"<span style='color:{color};font-weight:600'>{status}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.metric(label, f"{actual_val:.1f}{unit} / {goal_val:.1f}{unit}" if goal_val > 0 else f"{actual_val:.1f}{unit}")

            st.caption(f"{pct:.1f}% of goal" if goal_val > 0 else "No goal set")
            if goal_val > 0:
                st.progress(min(pct / 100.0, 1.0))


def render_dashboard(daily_df: pd.DataFrame, macro_df: pd.DataFrame, active_goal: dict | None) -> None:
    st.subheader("Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    lw = latest_metric(daily_df, "weight_lbs")
    lbf = latest_metric(daily_df, "body_fat_pct")
    ls = latest_metric(daily_df, "steps")
    lc = latest_metric(macro_df, "cals")

    c1.metric("Latest Weight", "-" if lw is None else f"{lw:.1f} lbs")
    c2.metric("Latest Body Fat", "-" if lbf is None else f"{lbf:.1f}%")
    c3.metric("Latest Steps", "-" if ls is None else f"{int(ls):,}")
    c4.metric("Latest Food Cals", "-" if lc is None else f"{int(lc):,}")

    compare_date = st.date_input("Goal Comparison Date", value=date.today(), key="dashboard_compare_date")
    if active_goal:
        actual = totals_for_date(macro_df, compare_date)
        goal_name = active_goal.get("goal_name", "Current Goal")
        render_goals_vs_actual(actual, active_goal, f"{goal_name}: {compare_date.isoformat()}")
    else:
        st.info("No saved macro goal yet. Save one from Macro Calculator to enable goal comparison.")

    if not daily_df.empty:
        for col in ["weight_lbs", "body_fat_pct", "steps"]:
            plot_df = daily_df[["date", col]].dropna()
            if not plot_df.empty:
                fig = px.line(plot_df, x="date", y=col, markers=True, title=col.replace("_", " ").title())
                st.plotly_chart(fig, use_container_width=True)

    if not macro_df.empty:
        for col in ["cals", "protein", "carbs", "fat"]:
            fig = px.line(macro_df, x="date", y=col, markers=True, title=f"Daily {col.title()} Total")
            st.plotly_chart(fig, use_container_width=True)

    if daily_df.empty and macro_df.empty:
        st.info("No data yet. Start in Food Log and Body Metrics.")


def render_food_log(client: Client, foods_df: pd.DataFrame, macro_df: pd.DataFrame, active_goal: dict | None) -> None:
    st.subheader("Food Log")
    log_date = st.date_input("Date", value=date.today(), key="food_log_date")

    if foods_df.empty:
        st.warning("No foods available. Add foods in Manage Foods first.")
        return

    foods_df = foods_df.copy()

    def label_food(r: pd.Series) -> str:
        if r["input_mode"] == "quantity":
            return f"{r['name']} ({r['category']}) - {r['units_per_serving']} {r['unit_label']}(s)/serving"
        return f"{r['name']} ({r['category']}) - {r['grams_per_serving']}g serving"

    foods_df["label"] = foods_df.apply(label_food, axis=1)
    selected = st.selectbox("Food", options=foods_df["label"].tolist())
    selected_food = foods_df.loc[foods_df["label"] == selected].iloc[0]

    if selected_food["input_mode"] == "quantity":
        amount = st.number_input(
            f"Quantity eaten ({selected_food['unit_label']})",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )
    else:
        amount = st.number_input("Grams eaten", min_value=0.0, value=0.0, step=1.0)

    notes = st.text_input("Notes (optional)")

    if st.button("Add Food"):
        if amount <= 0:
            st.error("Amount must be greater than 0.")
        else:
            food_id = int(selected_food["id"])
            grams_to_log = float(amount)
            if selected_food["input_mode"] == "quantity":
                units_per_serving = float(selected_food["units_per_serving"] or 1)
                servings = float(amount) / units_per_serving
                grams_to_log = servings * float(selected_food["grams_per_serving"])

            log_food_entry(client, log_date, food_id, grams_to_log, notes)
            st.success("Food entry added.")
            st.rerun()

    logs_df = fetch_food_logs(client, log_date)

    st.markdown("### Running Totals")
    if logs_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calories", "0")
        c2.metric("Protein", "0 g")
        c3.metric("Carbs", "0 g")
        c4.metric("Fat", "0 g")
        st.info("No foods logged for this date yet.")
        return

    total_cals = logs_df["cals"].sum()
    total_pro = logs_df["protein"].sum()
    total_carb = logs_df["carbs"].sum()
    total_fat = logs_df["fat"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calories", f"{total_cals:.0f}")
    c2.metric("Protein", f"{total_pro:.1f} g")
    c3.metric("Carbs", f"{total_carb:.1f} g")
    c4.metric("Fat", f"{total_fat:.1f} g")

    if active_goal:
        actual = {"cals": total_cals, "protein": total_pro, "carbs": total_carb, "fat": total_fat}
        goal_name = active_goal.get("goal_name", "Current Goal")
        render_goals_vs_actual(actual, active_goal, f"{goal_name}: {log_date.isoformat()}")
    else:
        st.info("No saved macro goal yet. Save one from Macro Calculator to compare daily totals.")

    st.markdown("### Entries")
    display = logs_df.copy()
    for col in ["grams", "servings", "cals", "protein", "carbs", "fat"]:
        display[col] = display[col].round(2)
    st.dataframe(display, use_container_width=True)

    delete_options = {f"#{int(r.id)} - {r.name} ({r.servings:.2f} servings)": int(r.id) for _, r in logs_df.iterrows()}
    selected_delete = st.selectbox("Delete an entry", options=["None"] + list(delete_options.keys()))
    if selected_delete != "None" and st.button("Delete Selected Entry"):
        delete_food_log(client, delete_options[selected_delete])
        st.success("Entry deleted.")
        st.rerun()


def render_manage_foods(client: Client, foods_df: pd.DataFrame) -> None:
    st.subheader("Manage Foods")
    st.caption("Add custom foods with macros per serving. These become selectable in Food Log.")

    with st.form("add_food_form"):
        name = st.text_input("Food name")
        category = st.selectbox("Category", ["Protein", "Carbs", "Fats", "Other"])
        input_mode_label = st.selectbox("Input mode", ["Grams", "Quantity"])

        grams_per_serving = st.number_input("Grams per serving", min_value=0.1, value=100.0, step=1.0)
        cals = st.number_input("Calories per serving", min_value=0.0, value=0.0, step=1.0)
        pro = st.number_input("Protein per serving (g)", min_value=0.0, value=0.0, step=0.5)
        carbs = st.number_input("Carbs per serving (g)", min_value=0.0, value=0.0, step=0.5)
        fat = st.number_input("Fat per serving (g)", min_value=0.0, value=0.0, step=0.5)

        input_mode = "quantity" if input_mode_label == "Quantity" else "grams"
        unit_label = "serving"
        units_per_serving = 1.0

        if input_mode == "quantity":
            unit_label = st.text_input("Quantity unit label", value="item", help="Examples: egg, scoop, slice")
            units_per_serving = st.number_input(
                "Units per serving",
                min_value=0.1,
                value=1.0,
                step=0.5,
                help="If macros are for 2 eggs, set this to 2.",
            )

        submit = st.form_submit_button("Add Food")

    if submit:
        if not name.strip():
            st.error("Food name is required.")
        else:
            try:
                add_food(client, name, category, grams_per_serving, cals, pro, carbs, fat, input_mode, unit_label, units_per_serving)
                st.success(f"Added '{name}'.")
                st.rerun()
            except Exception as exc:
                msg = str(exc)
                if "duplicate key" in msg.lower() or "23505" in msg:
                    st.error("A food with that name already exists.")
                else:
                    st.error(f"Could not add food: {msg}")

    st.markdown("### Current Food Library")
    show_cols = [
        "name",
        "category",
        "input_mode",
        "unit_label",
        "units_per_serving",
        "grams_per_serving",
        "cals_per_serving",
        "protein_per_serving",
        "carbs_per_serving",
        "fat_per_serving",
    ]
    display = foods_df[show_cols].copy() if not foods_df.empty else foods_df
    st.dataframe(display, use_container_width=True)


def render_add_entry(client: Client) -> None:
    st.subheader("Body Metrics")
    st.caption("One row per day. Saving a date again updates that day.")

    with st.form("entry_form"):
        entry_date = st.date_input("Date", value=date.today(), key="body_date")
        weight_raw = st.text_input("Weight (lbs)", placeholder="e.g., 188.4")
        body_fat_raw = st.text_input("Body Fat (%)", placeholder="e.g., 17.9")
        steps_raw = st.text_input("Steps", placeholder="e.g., 10421")
        calories_raw = st.text_input("Manual calories (optional)", placeholder="If you want to override/track separately")
        food_notes = st.text_area("Food Notes", placeholder="Extra note for the day")

        submitted = st.form_submit_button("Save Body Metrics")

    if submitted:
        weight_lbs = parse_optional_float(weight_raw)
        body_fat_pct = parse_optional_float(body_fat_raw)
        steps = parse_optional_int(steps_raw)
        calories = parse_optional_int(calories_raw)

        invalid = []
        if weight_raw.strip() and weight_lbs is None:
            invalid.append("Weight")
        if body_fat_raw.strip() and body_fat_pct is None:
            invalid.append("Body Fat")
        if steps_raw.strip() and steps is None:
            invalid.append("Steps")
        if calories_raw.strip() and calories is None:
            invalid.append("Manual Calories")
        if body_fat_pct is not None and not (0 <= body_fat_pct <= 100):
            invalid.append("Body Fat (must be 0 to 100)")

        if invalid:
            st.error(f"Please correct invalid fields: {', '.join(invalid)}")
            return

        upsert_daily_entry(client, entry_date, weight_lbs, body_fat_pct, steps, calories, food_notes)
        st.success(f"Saved body metrics for {entry_date.isoformat()}.")


def render_macro_calculator(client: Client, daily_df: pd.DataFrame, active_goal: dict | None) -> None:
    st.subheader("Macro Calculator")
    st.caption("Weight-based calorie and macro targets for maintenance and deficit.")

    if active_goal:
        st.success(
            "Saved Goal: "
            f"{active_goal.get('goal_name', 'Current Goal')} | "
            f"Cals {float(active_goal.get('target_cals', 0)):.0f}, "
            f"Protein {float(active_goal.get('target_protein', 0)):.1f}g, "
            f"Carbs {float(active_goal.get('target_carbs', 0)):.1f}g, "
            f"Fat {float(active_goal.get('target_fat', 0)):.1f}g"
        )
    else:
        st.info("No macro goal saved yet. Use one of the save buttons below.")

    latest_weight = latest_metric(daily_df, "weight_lbs")
    latest_bf = latest_metric(daily_df, "body_fat_pct")
    default_weight = float(latest_weight) if latest_weight is not None else 189.0
    default_bf = float(latest_bf) if latest_bf is not None else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        body_weight = st.number_input("Body Weight (lbs)", min_value=1.0, value=default_weight, step=1.0)
        body_fat_pct = st.number_input("Body Fat (%)", min_value=0.0, max_value=100.0, value=default_bf, step=0.1)
    with c2:
        bmr = st.number_input("BMR", min_value=0.0, value=2000.0, step=10.0)
        daily_activity = st.number_input("Daily Activity Cals", min_value=0.0, value=600.0, step=10.0)
    with c3:
        maintenance_cals = st.number_input("Maintenance Calories", min_value=0.0, value=float(bmr + daily_activity), step=10.0)
        deficit_delta = st.number_input("Deficit Amount", min_value=0.0, value=300.0, step=10.0)

    estimated_tee = bmr + daily_activity
    deficit_cals = max(0.0, maintenance_cals - deficit_delta)

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated TEE", f"{estimated_tee:.0f}")
    m2.metric("Maintenance Target", f"{maintenance_cals:.0f}")
    m3.metric("Deficit Target", f"{deficit_cals:.0f}")

    if body_fat_pct > 0:
        lean_mass = body_weight * (1 - body_fat_pct / 100)
        fat_mass = body_weight * (body_fat_pct / 100)
        lm1, lm2 = st.columns(2)
        lm1.metric("Lean Mass", f"{lean_mass:.1f} lbs")
        lm2.metric("Fat Mass", f"{fat_mass:.1f} lbs")

    st.markdown("### Weight Multiplier Tables")
    needed_cals_df = pd.DataFrame({"cals_per_lb": [14, 15, 16, 17], "needed_cals": [body_weight * x for x in [14, 15, 16, 17]]})
    protein_df = pd.DataFrame({"protein_ratio_g_per_lb": [1.0, 1.1, 1.25, 1.5], "protein_grams": [body_weight * x for x in [1.0, 1.1, 1.25, 1.5]]})
    carb_df = pd.DataFrame({"carb_ratio_g_per_lb": [0.75, 1.0, 1.25, 1.5], "carb_grams": [body_weight * x for x in [0.75, 1.0, 1.25, 1.5]]})
    fat_df = pd.DataFrame({"fat_ratio_g_per_lb": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70], "fat_grams": [body_weight * x for x in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]]})

    t1, t2 = st.columns(2)
    with t1:
        st.dataframe(needed_cals_df.round(2), use_container_width=True)
        st.dataframe(protein_df.round(2), use_container_width=True)
    with t2:
        st.dataframe(carb_df.round(2), use_container_width=True)
        st.dataframe(fat_df.round(2), use_container_width=True)

    st.markdown("### Macro Targets")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### Maintenance Plan")
        protein_ratio_maint = st.number_input("Protein g/lb (Maintenance)", min_value=0.0, value=1.25, step=0.05)
        fat_ratio_maint = st.number_input("Fat g/lb (Maintenance)", min_value=0.0, value=0.60, step=0.05)
    with p2:
        st.markdown("#### Deficit Plan")
        protein_ratio_def = st.number_input("Protein g/lb (Deficit)", min_value=0.0, value=1.25, step=0.05)
        fat_ratio_def = st.number_input("Fat g/lb (Deficit)", min_value=0.0, value=0.50, step=0.05)

    maint = macro_breakdown(maintenance_cals, body_weight, protein_ratio_maint, fat_ratio_maint)
    cut = macro_breakdown(deficit_cals, body_weight, protein_ratio_def, fat_ratio_def)

    out1, out2 = st.columns(2)
    with out1:
        st.markdown("#### Maintenance Results")
        if st.button("Save Maintenance As Active Goal"):
            try:
                save_active_macro_goal(
                    client,
                    "Maintenance",
                    maintenance_cals,
                    maint["protein_g"],
                    maint["carbs_g"],
                    maint["fat_g"],
                    body_weight,
                    body_fat_pct,
                    bmr,
                    daily_activity,
                )
                st.success("Maintenance goal saved.")
                st.rerun()
            except Exception as exc:
                st.error("Could not save goal. Add macro_goals table in Supabase SQL Editor first.")
                st.code(str(exc))
        st.dataframe(
            pd.DataFrame(
                [
                    {"metric": "Grams Protein", "value": maint["protein_g"]},
                    {"metric": "Cals Protein", "value": maint["protein_cals"]},
                    {"metric": "Grams Fat", "value": maint["fat_g"]},
                    {"metric": "Cals Fat", "value": maint["fat_cals"]},
                    {"metric": "Protein + Fat Cals", "value": maint["pf_cals"]},
                    {"metric": "Needed Cals", "value": maintenance_cals},
                    {"metric": "Grams Carbs", "value": maint["carbs_g"]},
                    {"metric": "Cals Carbs", "value": maint["carb_cals"]},
                ]
            ).round(2),
            use_container_width=True,
            hide_index=True,
        )
    with out2:
        st.markdown("#### Deficit Results")
        if st.button("Save Deficit As Active Goal"):
            try:
                save_active_macro_goal(
                    client,
                    "Deficit",
                    deficit_cals,
                    cut["protein_g"],
                    cut["carbs_g"],
                    cut["fat_g"],
                    body_weight,
                    body_fat_pct,
                    bmr,
                    daily_activity,
                )
                st.success("Deficit goal saved.")
                st.rerun()
            except Exception as exc:
                st.error("Could not save goal. Add macro_goals table in Supabase SQL Editor first.")
                st.code(str(exc))
        st.dataframe(
            pd.DataFrame(
                [
                    {"metric": "Grams Protein", "value": cut["protein_g"]},
                    {"metric": "Cals Protein", "value": cut["protein_cals"]},
                    {"metric": "Grams Fat", "value": cut["fat_g"]},
                    {"metric": "Cals Fat", "value": cut["fat_cals"]},
                    {"metric": "Protein + Fat Cals", "value": cut["pf_cals"]},
                    {"metric": "Needed Cals", "value": deficit_cals},
                    {"metric": "Grams Carbs", "value": cut["carbs_g"]},
                    {"metric": "Cals Carbs", "value": cut["carb_cals"]},
                ]
            ).round(2),
            use_container_width=True,
            hide_index=True,
        )

    if maint["remaining_cals"] < 0 or cut["remaining_cals"] < 0:
        st.warning("Protein and fat calories exceed total target calories in at least one plan. Increase calories or lower ratios.")


def render_import(client: Client) -> None:
    st.subheader("Import CSV (Body Metrics)")
    st.caption("Upload CSV with columns like: date, weight_lbs, body_fat_pct, steps, calories, food_notes")

    upload = st.file_uploader("Choose CSV", type=["csv"])
    if upload is None:
        return

    raw = pd.read_csv(upload)
    cleaned = normalize_import_columns(raw)

    st.write("Preview (after column mapping):")
    st.dataframe(cleaned.head(20), use_container_width=True)

    if st.button("Import Rows"):
        count = import_rows(client, cleaned)
        st.success(f"Imported/updated {count} row(s).")


def render_export(daily_df: pd.DataFrame, macro_df: pd.DataFrame) -> None:
    st.subheader("Export")
    if daily_df.empty and macro_df.empty:
        st.info("No data to export yet.")
        return

    if not daily_df.empty:
        out_daily = daily_df.copy()
        out_daily["date"] = out_daily["date"].dt.date.astype(str)
        st.download_button(
            label="Download Body Metrics CSV",
            data=out_daily.to_csv(index=False).encode("utf-8"),
            file_name="body_metrics_export.csv",
            mime="text/csv",
        )

    if not macro_df.empty:
        out_macro = macro_df.copy()
        out_macro["date"] = out_macro["date"].dt.date.astype(str)
        st.download_button(
            label="Download Daily Macro Totals CSV",
            data=out_macro.to_csv(index=False).encode("utf-8"),
            file_name="daily_macro_totals_export.csv",
            mime="text/csv",
        )


def main() -> None:
    st.set_page_config(page_title="Fitness Tracker", layout="wide")
    st.title("Personal Fitness Tracker")
    st.caption("Supabase-backed tracking for food, macros, and body metrics across devices.")

    client = require_supabase()
    if client is None:
        st.stop()

    try:
        seed_default_foods(client)
    except Exception as exc:
        st.error("Supabase tables are not ready or credentials are invalid.")
        st.code(str(exc))
        st.info("Create tables in Supabase first, then refresh.")
        st.stop()

    foods_df = fetch_foods(client)
    daily_df = fetch_daily_entries(client)
    macro_df = fetch_daily_macro_totals(client)
    active_goal = fetch_active_macro_goal(client)

    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Food Log", "Manage Foods", "Body Metrics", "Macro Calculator", "Import CSV", "Export"],
    )

    if page == "Dashboard":
        render_dashboard(daily_df, macro_df, active_goal)
    elif page == "Food Log":
        render_food_log(client, foods_df, macro_df, active_goal)
    elif page == "Manage Foods":
        render_manage_foods(client, foods_df)
    elif page == "Body Metrics":
        render_add_entry(client)
    elif page == "Macro Calculator":
        render_macro_calculator(client, daily_df, active_goal)
    elif page == "Import CSV":
        render_import(client)
    elif page == "Export":
        render_export(daily_df, macro_df)


if __name__ == "__main__":
    main()
