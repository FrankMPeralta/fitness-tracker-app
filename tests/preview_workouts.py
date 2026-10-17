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
    client = st.session_state['_db']
    template = next(t for t in client.tables['workout_templates'] if t['name'] == 'Lower Body A')
    baseline = dict(id='preview-baseline', name=template['name'], performed_date='2026-10-12',
                    status='completed', prescription=template['exercises'], knee_discomfort=0)
    sets = [dict(exercise_id=item['exercise_id'], exercise_position=i, set_number=n,
                 weight_lbs=100 if item['rep_max'] else 0, reps=item['rep_max'], rir=2, completed=True, notes='')
            for i, item in enumerate(template['exercises']) for n in range(1, item['set_count'] + 1)]
    client.rpc('save_workout', dict(p_session=baseline, p_sets=sets)).execute()
render_workouts(st.session_state['_db'], 'user-a', date(2026, 10, 19))
