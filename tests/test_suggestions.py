import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from workout_suggestions import suggest_weight, weight_label


ITEM = dict(exercise_id='bench', name='Bench Press', set_count=3, rep_min=5, rep_max=8)


def history(weights, reps=None, effort=None, item=None, knee=0):
    item = item or ITEM
    reps = reps or [8] * len(weights)
    effort = effort or [2] * len(weights)
    return [dict(exercise_id=item['exercise_id'], weight_lbs=w, reps=r, rir=e,
                 completed=True, notes='', workout_sessions=dict(knee_discomfort=knee, prescription=[item]))
            for w, r, e in zip(weights, reps, effort)]


class SuggestionTests(unittest.TestCase):
    def test_missing_history_does_not_invent_weight(self):
        self.assertIsNone(suggest_weight(ITEM, [])['weight'])

    def test_repeated_working_weight_wins_over_light_ramp(self):
        suggestion = suggest_weight(ITEM, history([90, 100, 100], [8, 6, 6], [6, 2, 2]))
        self.assertEqual(suggestion['weight'], 100)

    def test_single_heavier_top_set_is_reduced_for_repeated_sets(self):
        self.assertEqual(suggest_weight(ITEM, history([70, 80, 100], effort=[8, 5, 2]))['weight'], 90)
        self.assertEqual(suggest_weight(ITEM, history([80, 80, 100], effort=[5, 4, 2]))['weight'], 90)

    def test_full_top_range_with_reserve_allows_small_increment(self):
        self.assertEqual(suggest_weight(ITEM, history([100, 100, 100]))['weight'], 105)

    def test_failure_reduces_load(self):
        self.assertEqual(suggest_weight(ITEM, history([100] * 3, effort=[2, 1, 0]))['weight'], 90)

    def test_missing_rir_does_not_allow_increase(self):
        self.assertEqual(suggest_weight(ITEM, history([100] * 3, effort=[None] * 3))['weight'], 100)

    def test_partial_session_or_new_rep_target_does_not_allow_increase(self):
        self.assertEqual(suggest_weight(ITEM, history([100] * 2))['weight'], 100)
        self.assertEqual(suggest_weight(dict(ITEM, rep_max=10), history([100] * 3, reps=[10] * 3))['weight'], 100)

    def test_uncompleted_sets_are_ignored(self):
        sets = history([100, 100, 200], [6] * 3)
        sets[-1]['completed'] = False
        self.assertEqual(suggest_weight(ITEM, sets)['weight'], 100)

    def test_knee_discomfort_blocks_squat_increase(self):
        item = dict(ITEM, name='Front Squat')
        suggestion = suggest_weight(item, history([100] * 3, item=item, knee=3))
        self.assertEqual(suggestion['weight'], 100)
        self.assertIn('3/10', suggestion['caution'])

    def test_bodyweight_and_timed_work_do_not_auto_increase(self):
        self.assertEqual(suggest_weight(ITEM, history([0] * 3))['weight'], 0)
        timed = dict(ITEM, rep_min=None, rep_max=None)
        self.assertEqual(suggest_weight(timed, history([60] * 3, item=timed))['weight'], 60)

    def test_custom_override_and_increment(self):
        self.assertEqual(suggest_weight(dict(ITEM, suggested_weight_lbs=85), [])['weight'], 85)
        self.assertEqual(suggest_weight(dict(ITEM, weight_increment_lbs=2.5), history([100] * 3))['weight'], 102.5)

    def test_large_dumbbell_jump_is_not_suggested(self):
        self.assertEqual(suggest_weight(ITEM, history([20] * 3), 'Dumbbells')['weight'], 20)

    def test_dumbbell_units_and_cable_notes(self):
        sets = history([100] * 3, [6] * 3)
        self.assertIn('per dumbbell', weight_label(suggest_weight(ITEM, sets, 'Dumbbells')))
        for entry in sets:
            entry['notes'] = 'Cable'
        self.assertNotIn('per dumbbell', weight_label(suggest_weight(ITEM, sets, 'Dumbbells')))

    def test_recovery_week_never_increases_load(self):
        self.assertEqual(suggest_weight(ITEM, history([100] * 3), allow_increase=False)['weight'], 100)

    def test_actual_weights_remain_blank_until_user_applies_suggestion(self):
        script = '''
import streamlit as st
from workouts import render_logger
from tests.fake_supabase import FakeClient
render_logger(FakeClient(), 'user-a', [{'id':'bench','name':'Bench Press','equipment':'Barbell'}],
    dict(id='new-session', name='Upper A', performed_date='2026-10-19', prescription=[
        dict(exercise_id='bench', name='Bench Press', set_count=3, rep_min=5, rep_max=8)]))
'''
        with patch('workouts.previous_sets', return_value=('2026-10-12', history([100] * 3, [6] * 3))):
            app = AppTest.from_string(script).run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.number_input[0].value)
            self.assertTrue(any('100 lb' in m.value for m in app.markdown))
            app.number_input[0].set_value(95)
            next(b for b in app.button if b.label == 'Use Suggested Weights').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.number_input[0].value, 95)
            self.assertEqual(app.number_input[3].value, 100)
            self.assertIsNone(app.number_input[1].value)
            self.assertFalse(any(c.value for c in app.checkbox))

    def test_saved_baseline_flows_through_logger(self):
        app = AppTest.from_file('tests/preview_workouts.py').run()
        next(b for b in app.button if b.label == 'Schedule This Week').click().run()
        next(b for b in app.button if b.label == 'Open Workout').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any('105 lb' in m.value for m in app.markdown))
        next(b for b in app.button if b.label == 'Use Suggested Weights').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.number_input[0].value, 105)

    def test_substitution_drops_old_override_and_can_fill_new_blank_sets(self):
        script = '''
from workouts import render_logger
from tests.fake_supabase import FakeClient
client = FakeClient()
client.tables['workout_sets'] = [dict(id=str(n), user_id='user-a', session_id='editing',
    exercise_id='bench', exercise_position=0, set_number=n, weight_lbs=90,
    reps=6, rir=2, completed=True, notes='') for n in range(1,4)]
render_logger(client, 'user-a', [dict(id='bench', name='Bench Press', equipment='Barbell'),
    dict(id='other', name='Another Exercise', equipment='Barbell')],
    dict(id='editing', name='Upper A', performed_date='2026-10-19', saved=True,
         prescription=[dict(exercise_id='bench', name='Bench Press', set_count=3,
             rep_min=5, rep_max=8, suggested_weight_lbs=500)]))
'''
        with patch('workouts.previous_sets', return_value=('2026-10-12', history([100] * 3, [6] * 3))):
            app = AppTest.from_string(script).run()
            next(s for s in app.selectbox if s.label == 'Exercise 1').set_value('other').run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.number_input[0].value)
            self.assertFalse(any('500 lb' in m.value for m in app.markdown))
            next(b for b in app.button if b.label == 'Use Suggested Weights').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.number_input[0].value, 100)
