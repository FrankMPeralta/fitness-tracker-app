"""Local UI preview with synthetic data only; not the deployed entry point."""
from datetime import date
import streamlit as st
from tests.fake_supabase import FakeClient
from workouts import render_workouts, seed_starters

st.set_page_config(page_title='Workout Preview', layout='wide')
st.title('Personal Fitness Tracker')
if '_db' not in st.session_state:
    st.session_state['_db'] = FakeClient()
    seed_starters(st.session_state['_db'], 'user-a')
render_workouts(st.session_state['_db'], 'user-a', date(2026, 10, 12))
