import unittest
from datetime import date

from streamlit.testing.v1 import AppTest

from workouts import PROGRAM_END, STARTERS, previous_sets, program_label, seed_starters, validate_sets
from tests.fake_supabase import FakeClient

SCRIPT = '''
import streamlit as st
from datetime import date
from workouts import render_workouts, seed_starters
from tests.fake_supabase import FakeClient
if '_db' not in st.session_state:
    st.session_state['_db'] = FakeClient()
    seed_starters(st.session_state['_db'], 'user-a')
render_workouts(st.session_state['_db'], 'user-a', date(2026, 10, 12))
'''


def button(app, label):
    return next(b for b in app.button if b.label == label)


class WorkoutTests(unittest.TestCase):
    def test_previous_performance_query(self):
        import httpx
        from postgrest import SyncPostgrestClient
        seen = []

        def response(request):
            seen.append(request.url.params)
            return httpx.Response(200, json=[
                dict(session_id='older', set_number=1, workout_sessions={'performed_date': '2026-10-01'}),
                dict(session_id='latest', set_number=2, workout_sessions={'performed_date': '2026-10-08'}),
                dict(session_id='latest', set_number=1, workout_sessions={'performed_date': '2026-10-08'}),
            ])

        with httpx.Client(transport=httpx.MockTransport(response)) as transport:
            client = SyncPostgrestClient('https://example.test', http_client=transport)
            day, sets = previous_sets(client, 'user-a', 'exercise', date(2026, 10, 12), 'current')
        self.assertEqual(day, '2026-10-08')
        self.assertEqual([s['set_number'] for s in sets], [1, 2])
        self.assertEqual(seen[0]['user_id'], 'eq.user-a')
        self.assertEqual(seen[0]['completed'], 'eq.True')
        self.assertEqual(seen[0]['session_id'], 'neq.current')
        self.assertEqual(seen[0]['workout_sessions.status'], 'eq.completed')
        self.assertIn('knee_discomfort', seen[0]['select'])
        self.assertEqual(seen[0]['order'], 'workout_sessions(performed_date).desc')

    def test_missing_migration_is_reported_without_crashing(self):
        app = AppTest.from_string('''
from datetime import date
from workouts import render_workouts
class MissingDatabase:
    def table(self, name):
        raise RuntimeError('Table missing')
render_workouts(MissingDatabase(), 'user-a', date(2026, 10, 12))
''').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 1)
        self.assertIn('nutrition pages', app.error[0].value)

    def test_program_boundaries(self):
        self.assertIn('starts', program_label(date(2026, 10, 11)))
        self.assertIn('Week 1 of 16', program_label(date(2026, 10, 12)))
        self.assertIn('Recovery week', program_label(date(2026, 11, 2)))
        self.assertIn('Week 16 of 16', program_label(PROGRAM_END))
        self.assertIn('fat-loss', program_label(date(2027, 2, 1)))

    def test_seed_preserves_edits_and_is_user_scoped(self):
        client = FakeClient()
        seed_starters(client, 'user-a')
        client.tables['workout_templates'][0]['exercises'][0]['set_count'] = 2
        seed_starters(client, 'user-a')
        self.assertEqual(len(client.tables['workout_templates']), len(STARTERS))
        self.assertEqual(client.tables['workout_templates'][0]['exercises'][0]['set_count'], 2)
        seed_starters(client, 'user-b')
        self.assertEqual(len(client.tables['workout_templates']), len(STARTERS) * 2)

    def test_completed_set_validation_and_timed_work(self):
        item = dict(exercise_id='exercise', name='Squat', set_count=1, rep_min=5, rep_max=6)
        entry = dict(exercise_id='exercise', exercise_position=0, set_number=1, weight_lbs=None, reps=None, rir=None, completed=False)
        validate_sets([entry], False, [item])
        with self.assertRaises(ValueError):
            validate_sets([entry], True, [item])
        entry['completed'] = True
        with self.assertRaises(ValueError):
            validate_sets([entry], True, [item])
        entry.update(reps=6, rir=2)
        validate_sets([entry], True, [item])
        entry['rir'] = 11
        with self.assertRaises(ValueError):
            validate_sets([entry], True, [item])
        entry.update(reps=None, rir=None)
        item.update(rep_min=None, rep_max=None)
        validate_sets([entry], True, [item])

    def app_with_log(self):
        app = AppTest.from_string(SCRIPT, default_timeout=10).run()
        button(app, 'Schedule This Week').click().run()
        button(app, 'Open Workout').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        return app

    def test_save_reopen_and_edit_same_session(self):
        app = self.app_with_log()
        app.number_input[0].set_value(155.0)
        app.number_input[1].set_value(6)
        next(c for c in app.checkbox if (c.key or '').endswith('_1_done')).check()
        next(c for c in app.checkbox if c.label == 'Workout completed').check()
        button(app, 'Save Workout').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        db = app.session_state['_db']
        self.assertEqual(len(db.tables['workout_sessions']), 1)
        self.assertEqual(db.tables['workout_sessions'][0]['status'], 'completed')
        sid = db.tables['workout_sessions'][0]['id']
        app.radio[0].set_value('History').run()
        self.assertEqual(app.number_input[0].value, 155.0)
        app.number_input[0].set_value(160.0)
        button(app, 'Save Workout').click().run()
        self.assertEqual(db.tables['workout_sessions'][0]['id'], sid)
        self.assertEqual(len(db.tables['workout_sessions']), 1)
        self.assertEqual(db.tables['workout_sets'][0]['weight_lbs'], 160.0)

    def test_substitution_starts_blank(self):
        app = self.app_with_log()
        app.number_input[0].set_value(155.0)
        button(app, 'Save Workout').click().run()
        selector = next(s for s in app.selectbox if s.label == 'Exercise 1')
        other = next(e['id'] for e in app.session_state['_db'].tables['workout_exercises'] if e['name'] == 'Romanian Deadlift')
        selector.set_value(other).run()
        self.assertIsNone(app.number_input[0].value)
        self.assertFalse(next(c for c in app.checkbox if (c.key or '').endswith('_1_done')).value)

    def test_reschedule_then_regenerate_does_not_duplicate(self):
        app = AppTest.from_string(SCRIPT, default_timeout=10).run()
        button(app, 'Schedule This Week').click().run()
        db = app.session_state['_db']
        plan = db.tables['workout_plans'][0]
        next(d for d in app.date_input if d.key == 'move_date_' + plan['id']).set_value(date(2026, 10, 21))
        button(app, 'Move').click().run()
        button(app, 'Schedule This Week').click().run()
        self.assertEqual(len(db.tables['workout_plans']), 4)
        self.assertEqual(db.tables['workout_plans'][0]['scheduled_date'], '2026-10-21')

    def test_library_and_template_pages_render(self):
        app = AppTest.from_string(SCRIPT, default_timeout=10).run()
        for page in ('Templates', 'Exercise Library', 'History'):
            app.radio[0].set_value(page).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)


if __name__ == '__main__':
    unittest.main()
