import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path("fitness_tracker.db")

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


def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_entries (
                entry_date TEXT PRIMARY KEY,
                weight_lbs REAL,
                body_fat_pct REAL,
                steps INTEGER,
                calories INTEGER,
                food_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS foods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                grams_per_serving REAL NOT NULL,
                cals_per_serving REAL NOT NULL,
                protein_per_serving REAL NOT NULL,
                carbs_per_serving REAL NOT NULL,
                fat_per_serving REAL NOT NULL,
                input_mode TEXT NOT NULL DEFAULT 'grams',
                unit_label TEXT NOT NULL DEFAULT 'serving',
                units_per_serving REAL NOT NULL DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        ensure_foods_schema(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS food_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_date TEXT NOT NULL,
                food_id INTEGER NOT NULL,
                grams REAL NOT NULL,
                servings REAL NOT NULL,
                cals REAL NOT NULL,
                protein REAL NOT NULL,
                carbs REAL NOT NULL,
                fat REAL NOT NULL,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(food_id) REFERENCES foods(id)
            )
            """
        )


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


def ensure_foods_schema(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(foods)").fetchall()}
    if "input_mode" not in cols:
        conn.execute("ALTER TABLE foods ADD COLUMN input_mode TEXT NOT NULL DEFAULT 'grams'")
    if "unit_label" not in cols:
        conn.execute("ALTER TABLE foods ADD COLUMN unit_label TEXT NOT NULL DEFAULT 'serving'")
    if "units_per_serving" not in cols:
        conn.execute("ALTER TABLE foods ADD COLUMN units_per_serving REAL NOT NULL DEFAULT 1")

    # Set quantity-based defaults for common count-based foods.
    conn.execute(
        """
        UPDATE foods
        SET input_mode='quantity', unit_label='egg', units_per_serving=1
        WHERE lower(name)='eggs'
        """
    )
    conn.execute(
        """
        UPDATE foods
        SET input_mode='quantity', unit_label='serving', units_per_serving=1
        WHERE lower(name)='egg whites'
        """
    )
    conn.execute(
        """
        UPDATE foods
        SET input_mode='quantity', unit_label='scoop', units_per_serving=1
        WHERE lower(name) LIKE '%protein shake%'
        """
    )


def seed_default_foods() -> None:
    with get_conn() as conn:
        for food in DEFAULT_FOODS:
            conn.execute(
                """
                INSERT INTO foods (
                    name, category, grams_per_serving, cals_per_serving,
                    protein_per_serving, carbs_per_serving, fat_per_serving,
                    input_mode, unit_label, units_per_serving
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO NOTHING
                """,
                (
                    food["name"],
                    food["category"],
                    food["grams_per_serving"],
                    food["cals"],
                    food["protein"],
                    food["carbs"],
                    food["fat"],
                    default_food_input_config(food["name"])[0],
                    default_food_input_config(food["name"])[1],
                    default_food_input_config(food["name"])[2],
                ),
            )


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
    entry_date: date,
    weight_lbs: float | None,
    body_fat_pct: float | None,
    steps: int | None,
    calories: int | None,
    food_notes: str,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO daily_entries (entry_date, weight_lbs, body_fat_pct, steps, calories, food_notes)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_date) DO UPDATE SET
                weight_lbs=excluded.weight_lbs,
                body_fat_pct=excluded.body_fat_pct,
                steps=excluded.steps,
                calories=excluded.calories,
                food_notes=excluded.food_notes,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                entry_date.isoformat(),
                weight_lbs,
                body_fat_pct,
                steps,
                calories,
                food_notes,
            ),
        )


def fetch_daily_entries() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT entry_date as date, weight_lbs, body_fat_pct, steps, calories, food_notes
            FROM daily_entries
            ORDER BY entry_date
            """,
            conn,
        )

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_foods() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            """
            SELECT id, name, category, grams_per_serving, cals_per_serving,
                   protein_per_serving, carbs_per_serving, fat_per_serving,
                   input_mode, unit_label, units_per_serving
            FROM foods
            ORDER BY category, name
            """,
            conn,
        )


def log_food_entry(entry_date: date, food_id: int, grams: float, notes: str) -> None:
    with get_conn() as conn:
        food = conn.execute(
            """
            SELECT grams_per_serving, cals_per_serving, protein_per_serving,
                   carbs_per_serving, fat_per_serving
            FROM foods
            WHERE id = ?
            """,
            (food_id,),
        ).fetchone()

        if not food:
            raise ValueError("Food not found")

        grams_per_serving, cals_ps, pro_ps, carb_ps, fat_ps = food
        servings = grams / grams_per_serving if grams_per_serving > 0 else 0
        cals = servings * cals_ps
        pro = servings * pro_ps
        carbs = servings * carb_ps
        fat = servings * fat_ps

        conn.execute(
            """
            INSERT INTO food_logs (entry_date, food_id, grams, servings, cals, protein, carbs, fat, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (entry_date.isoformat(), food_id, grams, servings, cals, pro, carbs, fat, notes),
        )


def fetch_food_logs(entry_date: date) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT fl.id, f.name, f.category, fl.grams, fl.servings, fl.cals, fl.protein, fl.carbs, fl.fat, fl.notes
            FROM food_logs fl
            JOIN foods f ON f.id = fl.food_id
            WHERE fl.entry_date = ?
            ORDER BY fl.id DESC
            """,
            conn,
            params=(entry_date.isoformat(),),
        )
    return df


def delete_food_log(log_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM food_logs WHERE id = ?", (log_id,))


def fetch_daily_macro_totals() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT entry_date as date,
                   ROUND(SUM(cals), 2) AS cals,
                   ROUND(SUM(protein), 2) AS protein,
                   ROUND(SUM(carbs), 2) AS carbs,
                   ROUND(SUM(fat), 2) AS fat
            FROM food_logs
            GROUP BY entry_date
            ORDER BY entry_date
            """,
            conn,
        )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_food(
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
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO foods (
                name, category, grams_per_serving, cals_per_serving,
                protein_per_serving, carbs_per_serving, fat_per_serving,
                input_mode, unit_label, units_per_serving
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name.strip(),
                category,
                grams_per_serving,
                cals_per_serving,
                protein_per_serving,
                carbs_per_serving,
                fat_per_serving,
                input_mode,
                unit_label.strip() or "serving",
                units_per_serving,
            ),
        )


def import_rows(df: pd.DataFrame) -> int:
    imported = 0
    for _, row in df.iterrows():
        upsert_daily_entry(
            entry_date=row["date"],
            weight_lbs=None if pd.isna(row["weight_lbs"]) else float(row["weight_lbs"]),
            body_fat_pct=None if pd.isna(row["body_fat_pct"]) else float(row["body_fat_pct"]),
            steps=None if pd.isna(row["steps"]) else int(row["steps"]),
            calories=None if pd.isna(row["calories"]) else int(row["calories"]),
            food_notes=row["food_notes"],
        )
        imported += 1
    return imported


def latest_metric(df: pd.DataFrame, col: str):
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


def render_dashboard(daily_df: pd.DataFrame, macro_df: pd.DataFrame) -> None:
    st.subheader("Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    lw = latest_metric(daily_df, "weight_lbs") if not daily_df.empty else None
    lbf = latest_metric(daily_df, "body_fat_pct") if not daily_df.empty else None
    ls = latest_metric(daily_df, "steps") if not daily_df.empty else None
    lc = latest_metric(macro_df, "cals") if not macro_df.empty else None

    c1.metric("Latest Weight", "-" if lw is None else f"{lw:.1f} lbs")
    c2.metric("Latest Body Fat", "-" if lbf is None else f"{lbf:.1f}%")
    c3.metric("Latest Steps", "-" if ls is None else f"{int(ls):,}")
    c4.metric("Latest Food Cals", "-" if lc is None else f"{int(lc):,}")

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
        st.info("No data yet. Start in Food Log and Add / Edit Entry.")


def render_food_log(foods_df: pd.DataFrame) -> None:
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

    with st.form("food_log_form"):
        selected = st.selectbox("Food", options=foods_df["label"].tolist())
        selected_food = foods_df.loc[foods_df["label"] == selected].iloc[0]

        amount = 0.0
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
        submit = st.form_submit_button("Add Food")

    if submit:
        if amount <= 0:
            st.error("Amount must be greater than 0.")
        else:
            food_id = int(selected_food["id"])
            grams_to_log = float(amount)

            if selected_food["input_mode"] == "quantity":
                units_per_serving = float(selected_food["units_per_serving"] or 1)
                servings = float(amount) / units_per_serving
                grams_to_log = servings * float(selected_food["grams_per_serving"])

            log_food_entry(log_date, food_id, grams_to_log, notes)
            st.success("Food entry added.")

    logs_df = fetch_food_logs(log_date)

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

    st.markdown("### Entries")
    display = logs_df.copy()
    for col in ["grams", "servings", "cals", "protein", "carbs", "fat"]:
        display[col] = display[col].round(2)
    st.dataframe(display, use_container_width=True)

    delete_options = {f"#{int(r.id)} - {r.name} ({r.servings:.2f} servings)": int(r.id) for _, r in logs_df.iterrows()}
    selected_delete = st.selectbox("Delete an entry", options=["None"] + list(delete_options.keys()))
    if selected_delete != "None" and st.button("Delete Selected Entry"):
        delete_food_log(delete_options[selected_delete])
        st.success("Entry deleted. Refreshing...")
        st.rerun()


def render_manage_foods(foods_df: pd.DataFrame) -> None:
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
                add_food(name, category, grams_per_serving, cals, pro, carbs, fat, input_mode, unit_label, units_per_serving)
                st.success(f"Added '{name}'.")
                st.rerun()
            except sqlite3.IntegrityError:
                st.error("A food with that name already exists.")

    st.markdown("### Current Food Library")
    show_cols = [
        "name", "category", "input_mode", "unit_label", "units_per_serving",
        "grams_per_serving", "cals_per_serving", "protein_per_serving", "carbs_per_serving", "fat_per_serving",
    ]
    display = foods_df[show_cols].copy() if not foods_df.empty else foods_df
    st.dataframe(display, use_container_width=True)


def render_add_entry() -> None:
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

        upsert_daily_entry(
            entry_date=entry_date,
            weight_lbs=weight_lbs,
            body_fat_pct=body_fat_pct,
            steps=steps,
            calories=calories,
            food_notes=food_notes,
        )
        st.success(f"Saved body metrics for {entry_date.isoformat()}.")


def render_macro_calculator(daily_df: pd.DataFrame) -> None:
    st.subheader("Macro Calculator")
    st.caption("Weight-based calorie and macro targets for maintenance and deficit.")

    latest_weight = latest_metric(daily_df, "weight_lbs") if not daily_df.empty else None
    latest_bf = latest_metric(daily_df, "body_fat_pct") if not daily_df.empty else None
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
    needed_cals_df = pd.DataFrame(
        {
            "cals_per_lb": [14, 15, 16, 17],
            "needed_cals": [body_weight * x for x in [14, 15, 16, 17]],
        }
    )
    protein_df = pd.DataFrame(
        {
            "protein_ratio_g_per_lb": [1.0, 1.1, 1.25, 1.5],
            "protein_grams": [body_weight * x for x in [1.0, 1.1, 1.25, 1.5]],
        }
    )
    carb_df = pd.DataFrame(
        {
            "carb_ratio_g_per_lb": [0.75, 1.0, 1.25, 1.5],
            "carb_grams": [body_weight * x for x in [0.75, 1.0, 1.25, 1.5]],
        }
    )
    fat_df = pd.DataFrame(
        {
            "fat_ratio_g_per_lb": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
            "fat_grams": [body_weight * x for x in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]],
        }
    )

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


def render_import() -> None:
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
        count = import_rows(cleaned)
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
    st.caption("Log food by grams, keep running macro totals, and track body metrics.")

    init_db()
    seed_default_foods()

    foods_df = fetch_foods()
    daily_df = fetch_daily_entries()
    macro_df = fetch_daily_macro_totals()

    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Food Log", "Manage Foods", "Body Metrics", "Macro Calculator", "Import CSV", "Export"],
    )

    if page == "Dashboard":
        render_dashboard(daily_df, macro_df)
    elif page == "Food Log":
        render_food_log(foods_df)
    elif page == "Manage Foods":
        render_manage_foods(foods_df)
    elif page == "Body Metrics":
        render_add_entry()
    elif page == "Macro Calculator":
        render_macro_calculator(daily_df)
    elif page == "Import CSV":
        render_import()
    elif page == "Export":
        render_export(daily_df, macro_df)


if __name__ == "__main__":
    main()
