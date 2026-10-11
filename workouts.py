"""User-scoped workout planning and logging, independent of nutrition tables."""
from __future__ import annotations

from datetime import date, timedelta
from uuid import uuid4

import pandas as pd
import streamlit as st

PROGRAM_START = date(2026, 10, 12)
PROGRAM_END = date(2027, 1, 31)
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# Name, equipment, primary muscle group, sets, minimum/maximum reps, target note.
STARTERS = [
    ('Lower Body A', 0, [
        ('Back Squat', 'Barbell', 'Quadriceps', 3, 5, 6, ''),
        ('Romanian Deadlift', 'Barbell', 'Hamstrings', 3, 6, 8, ''),
        ('Reverse Lunges', 'Dumbbells', 'Quadriceps', 2, 8, 8, 'per leg'),
        ('Hamstring Curls', 'Machine', 'Hamstrings', 2, 10, 15, ''),
        ('Standing Calf Raises', 'Machine', 'Calves', 2, 12, 15, ''),
        ('Core Work', 'Bodyweight', 'Core', 3, None, None, 'Variable; record duration in notes'),
    ]),
    ('Upper Body A', 1, [
        ('Barbell Bench Press', 'Barbell', 'Chest', 3, 5, 6, ''),
        ('Pull-ups / Assisted Pull-ups', 'Pull-up bar', 'Back', 3, 6, 8, 'Record assistance in notes'),
        ('Incline Dumbbell Press', 'Dumbbells', 'Chest', 3, 8, 10, ''),
        ('Seated Cable Row', 'Cable', 'Back', 3, 8, 12, ''),
        ('Lateral Raises', 'Dumbbells', 'Shoulders', 2, 12, 15, ''),
        ('Triceps Pushdowns', 'Cable', 'Triceps', 2, 10, 15, ''),
    ]),
    ('Lower Body B', 3, [
        ('Trap-Bar / Conventional Deadlift', 'Barbell', 'Posterior chain', 3, 4, 6, ''),
        ('Front Squat / Goblet Squat', 'Barbell / Kettlebell', 'Quadriceps', 3, 8, 10, ''),
        ('Barbell Hip Thrust', 'Barbell', 'Glutes', 3, 8, 12, ''),
        ('Step-ups', 'Dumbbells', 'Quadriceps', 2, 8, 8, 'per leg'),
        ('Hamstring Curls', 'Machine', 'Hamstrings', 2, 10, 15, ''),
        ("Farmer's Carries", 'Dumbbells', 'Core', 3, None, None, 'Record duration or distance in notes'),
    ]),
    ('Upper Body B', 4, [
        ('Overhead Press', 'Barbell', 'Shoulders', 3, 5, 8, ''),
        ('Chest-Supported Row', 'Dumbbells', 'Back', 3, 6, 10, ''),
        ('Dumbbell Bench Press', 'Dumbbells', 'Chest', 3, 8, 12, ''),
        ('Lat Pulldown', 'Cable', 'Back', 3, 8, 12, ''),
        ('Rear Delt Fly', 'Dumbbells', 'Shoulders', 2, 12, 20, ''),
        ('Biceps Curls', 'Dumbbells', 'Biceps', 2, 10, 15, ''),
    ]),
    ('Optional Conditioning', 5, []),
]


def program_label(day: date) -> str:
    if day < PROGRAM_START:
        return 'Muscle growth and strength starts October 12'
    if day > PROGRAM_END:
        return 'Post-program / fat-loss phase'
    week = (day - PROGRAM_START).days // 7 + 1
    focus = ['Baseline', 'Progressive overload', 'Strength and volume', 'Consolidation'][(week - 1) // 4]
    if week in (4, 8, 12):
        focus = 'Recovery week'
    elif week == 16:
        focus = 'Performance assessment'
    return f'Muscle growth and strength · Week {week} of 16 · {focus}'


def rows(client, table: str, user_id: str, order: str = 'id', **filters) -> list[dict]:
    result = []
    offset = 0
    while True:
        query = client.table(table).select('*').eq('user_id', user_id).order(order)
        if order != 'id':
            query = query.order('id')
        for key, value in filters.items():
            query = query.eq(key, value)
        batch = query.range(offset, offset + 499).execute().data or []
        result.extend(batch)
        if len(batch) < 500:
            return result
        offset += 500


def seed_starters(client, user_id: str) -> None:
    exercises = {
        item[0]: {'user_id': user_id, 'name': item[0], 'equipment': item[1], 'muscle_group': item[2]}
        for _, _, items in STARTERS for item in items
    }
    client.table('workout_exercises').upsert(list(exercises.values()), on_conflict='user_id,name', ignore_duplicates=True).execute()
    library = {e['name']: e for e in rows(client, 'workout_exercises', user_id)}
    templates = []
    for name, weekday, items in STARTERS:
        prescription = [dict(exercise_id=library[i[0]]['id'], name=i[0], set_count=i[3],
                             rep_min=i[4], rep_max=i[5], rep_note=i[6]) for i in items]
        templates.append(dict(user_id=user_id, name=name, weekday=weekday, exercises=prescription))
    client.table('workout_templates').upsert(templates, on_conflict='user_id,name', ignore_duplicates=True).execute()


def validate_prescription(items: list[dict]) -> None:
    for item in items:
        if not item.get('exercise_id') or not item.get('name', '').strip():
            raise ValueError('Choose an exercise for each row.')
        if int(item['set_count']) != item['set_count'] or not 1 <= item['set_count'] <= 20:
            raise ValueError('Sets must be between 1 and 20.')
        low, high = item.get('rep_min'), item.get('rep_max')
        if (low is None) != (high is None) or (low is not None and not 1 <= low <= high <= 100):
            raise ValueError('Rep ranges must be between 1 and 100, or both blank for timed work.')


def validate_sets(sets: list[dict], completed: bool, items: list[dict]) -> None:
    validate_prescription(items)
    expected = {(i, n): item for i, item in enumerate(items) for n in range(1, item['set_count'] + 1)}
    if len(sets) != len(expected) or {(s['exercise_position'], s['set_number']) for s in sets} != set(expected):
        raise ValueError('The logged sets do not match the workout prescription.')
    for entry in sets:
        item = expected[entry['exercise_position'], entry['set_number']]
        if entry['exercise_id'] != item['exercise_id']:
            raise ValueError('The logged exercise does not match the workout.')
        for key in ('weight_lbs', 'reps', 'rir'):
            value = entry.get(key)
            if value is not None and (value < 0 or (key == 'rir' and value > 10)):
                raise ValueError('Weights and reps must be nonnegative; RIR must be between 0 and 10.')
        if entry['completed'] and item.get('rep_min') is not None and not entry.get('reps'):
            raise ValueError(f"Enter reps for each completed set of {item['name']}.")
        if entry['completed'] and entry.get('reps') == 0:
            raise ValueError('Completed sets need positive reps, or blank reps for timed work.')
    if completed and items and not any(s['completed'] for s in sets):
        raise ValueError('Mark at least one set completed before completing the workout.')


def previous_sets(client, user_id: str, exercise_id: str, before: date, session_id: str) -> tuple[str, list[dict]]:
    # Join only this exercise's completed sets, then locate the latest earlier session.
    result = client.table('workout_sets').select(
        '*,workout_sessions!inner(performed_date)'
    ).eq('user_id', user_id).eq('exercise_id', exercise_id).eq('completed', True).neq(
        'session_id', session_id
    ).lte('workout_sessions.performed_date', before.isoformat()).order(
        'workout_sessions(performed_date)', desc=True
    ).limit(1000).execute().data or []
    if not result:
        return '', []
    latest = max(result, key=lambda r: (r['workout_sessions']['performed_date'], r['session_id']))
    sets = [r for r in result if r['session_id'] == latest['session_id']]
    return latest['workout_sessions']['performed_date'], sorted(sets, key=lambda r: r['set_number'])


def render_logger(client, user_id: str, library: list[dict], session: dict) -> None:
    sid = session['id']
    key = f'workout_{user_id}_{sid}'
    st.markdown(f"#### {session['name']}")
    performed = st.date_input('Workout date', date.fromisoformat(session['performed_date']), key=f'{key}_date')
    st.caption(program_label(performed))
    existing = rows(client, 'workout_sets', user_id, session_id=sid) if session.get('saved') else []
    lookup = {(s['exercise_position'], s['set_number']): s for s in existing}
    library_by_id = {e['id']: e for e in library}
    prescription = []
    for i, original in enumerate(session['prescription']):
        item = dict(original)
        chosen = st.selectbox(f"Exercise {i + 1}", list(library_by_id),
                              index=list(library_by_id).index(item['exercise_id']),
                              format_func=lambda eid: library_by_id[eid]['name'], key=f'{key}_{i}_exercise')
        item.update(exercise_id=chosen, name=library_by_id[chosen]['name'])
        prescription.append(item)

    with st.form(f'{key}_form'):
        entries = []
        for i, item in enumerate(prescription):
            target = f"{item['rep_min']}-{item['rep_max']} reps" if item.get('rep_min') is not None else 'Timed / distance'
            with st.expander(f"{item['name']} · {item['set_count']} sets · {target}", expanded=(i == 0)):
                if item.get('rep_note'):
                    st.caption(item['rep_note'])
                prev_date, prev = previous_sets(client, user_id, item['exercise_id'], performed, sid)
                if prev:
                    st.caption('Previous (' + prev_date + '): ' + '; '.join(
                        f"{s['weight_lbs'] if s['weight_lbs'] is not None else 'BW / unrecorded'} lb × {s['reps'] if s['reps'] is not None else 'timed'}"
                        for s in prev))
                for number in range(1, item['set_count'] + 1):
                    old = lookup.get((i, number), {})
                    if old.get('exercise_id') != item['exercise_id']:
                        old = {}
                    # Include the exercise ID so a substitution starts with blank values.
                    prefix = f"{key}_{i}_{item['exercise_id']}_{number}"
                    st.markdown(f'**Set {number}**')
                    a, b = st.columns(2)
                    weight = a.number_input('Weight (lb)', min_value=0.0, value=float(old['weight_lbs']) if old.get('weight_lbs') is not None else None, step=2.5, key=f'{prefix}_weight')
                    reps = b.number_input('Reps', min_value=0, value=old.get('reps'), step=1, key=f'{prefix}_reps')
                    a, b = st.columns(2)
                    rir = a.number_input('RIR', min_value=0.0, max_value=10.0, value=float(old['rir']) if old.get('rir') is not None else None, step=0.5, key=f'{prefix}_rir')
                    done = b.checkbox('Completed', value=old.get('completed', False), key=f'{prefix}_done')
                    notes = st.text_input('Set notes', value=old.get('notes', ''), key=f'{prefix}_notes')
                    entries.append(dict(exercise_id=item['exercise_id'], exercise_position=i, set_number=number,
                                        weight_lbs=weight, reps=reps, rir=rir, completed=done, notes=notes))
        discomfort = st.selectbox('Knee discomfort', [None] + list(range(11)),
                                 index=([None] + list(range(11))).index(session.get('knee_discomfort')),
                                 format_func=lambda n: 'Not recorded' if n is None else f'{n} / 10', key=f'{key}_knee')
        notes = st.text_area('Workout / recovery notes', value=session.get('notes', ''), key=f'{key}_notes')
        finish = st.checkbox('Workout completed', value=session.get('status') == 'completed', key=f'{key}_finish')
        if st.form_submit_button('Save Workout', type='primary'):
            try:
                validate_sets(entries, finish, prescription)
                payload = dict(id=sid, planned_id=session.get('planned_id'), performed_date=performed.isoformat(),
                               name=session['name'], prescription=prescription,
                               status='completed' if finish else 'in_progress', knee_discomfort=discomfort, notes=notes)
                client.rpc('save_workout', {'p_session': payload, 'p_sets': entries}).execute()
                st.session_state['workout_notice'] = 'Workout saved.'
                st.rerun()
            except Exception as exc:
                st.error(f'Workout was not saved: {exc}')


def render_schedule(client, user_id: str, today: date, templates: list[dict], sessions: list[dict], library: list[dict]) -> None:
    selected = st.date_input('Week containing', today, key='workout_week')
    monday = selected - timedelta(days=selected.weekday())
    sunday = monday + timedelta(days=6)
    st.caption(program_label(selected))
    include_optional = st.checkbox('Include optional conditioning', value=False)
    if st.button('Schedule This Week', disabled=not templates):
        plans = [dict(user_id=user_id, template_id=t['id'], scheduled_date=(monday + timedelta(days=t['weekday'])).isoformat(),
                      schedule_key=f"{monday}:{t['id']}") for t in templates
                 if include_optional or t['name'] != 'Optional Conditioning']
        if plans:
            client.table('workout_plans').upsert(plans, on_conflict='user_id,schedule_key', ignore_duplicates=True).execute()
            st.rerun()
    plans = client.table('workout_plans').select('*').eq('user_id', user_id).gte(
        'scheduled_date', monday.isoformat()).lte('scheduled_date', sunday.isoformat()).order('scheduled_date').execute().data or []
    by_template = {t['id']: t for t in templates}
    by_plan = {s['planned_id']: s for s in sessions if s.get('planned_id')}
    for offset in range(7):
        day = monday + timedelta(days=offset)
        daily = [p for p in plans if p['scheduled_date'] == day.isoformat()]
        if not daily:
            st.write(f"**{DAYS[offset]} {day:%b %d}** · Recovery / unscheduled")
        for plan in daily:
            template = by_template[plan['template_id']]
            saved = by_plan.get(plan['id'])
            status = saved['status'].replace('_', ' ').title() if saved else 'Planned'
            st.write(f"**{DAYS[offset]} {day:%b %d}** · {template['name']} · {status}")
            if st.button('Open Workout', key=f"open_{plan['id']}"):
                st.session_state['workout_open_plan'] = plan['id']
            with st.expander('Move workout'):
                with st.form(f"move_{plan['id']}"):
                    moved = st.date_input('Scheduled date', day, key=f"move_date_{plan['id']}")
                    if st.form_submit_button('Move'):
                        client.table('workout_plans').update({'scheduled_date': moved.isoformat()}).eq('user_id', user_id).eq('id', plan['id']).execute()
                        st.rerun()
    with st.expander('Add a workout'):
        with st.form('add_workout_plan'):
            template_ids = list(by_template)
            chosen = st.selectbox('Template', template_ids, format_func=lambda tid: by_template[tid]['name'])
            day = st.date_input('Scheduled date', today, key='extra_workout_date')
            if st.form_submit_button('Add', disabled=not templates):
                client.table('workout_plans').insert(dict(user_id=user_id, template_id=chosen,
                    scheduled_date=day.isoformat(), schedule_key=str(uuid4()))).execute()
                st.rerun()
    opened = st.session_state.get('workout_open_plan')
    if opened:
        plan_rows = rows(client, 'workout_plans', user_id, id=opened)
        if plan_rows:
            plan = plan_rows[0]
            session = by_plan.get(opened)
            if session:
                session = dict(session, saved=True)
            else:
                template = by_template[plan['template_id']]
                draft_key = f'workout_draft_{user_id}_{opened}'
                sid = st.session_state.setdefault(draft_key, str(uuid4()))
                session = dict(id=sid, planned_id=opened, name=template['name'], performed_date=plan['scheduled_date'],
                               prescription=template['exercises'], status='in_progress')
            render_logger(client, user_id, library, session)


def render_templates(client, user_id: str, templates: list[dict], library: list[dict]) -> None:
    by_id = {t['id']: t for t in templates}
    selected = st.selectbox('Workout template', [None] + list(by_id), format_func=lambda tid: 'New template' if tid is None else by_id[tid]['name'])
    template = by_id.get(selected, dict(name='', weekday=0, exercises=[]))
    if not library:
        st.info('Add an exercise in Exercise Library first.')
        return
    key = selected or 'new'
    by_name = {e['name']: e for e in library}
    base = [dict(Exercise=i['name'], Sets=i['set_count'], Min=i.get('rep_min'), Max=i.get('rep_max'), Notes=i.get('rep_note', '')) for i in template['exercises']]
    with st.form(f'template_{key}'):
        name = st.text_input('Template name', template['name'])
        weekday = st.selectbox('Usual day', list(range(7)), index=template['weekday'], format_func=lambda n: DAYS[n])
        edited = st.data_editor(pd.DataFrame(base, columns=['Exercise', 'Sets', 'Min', 'Max', 'Notes']),
            num_rows='dynamic', hide_index=True, use_container_width=True, key=f'template_editor_{key}',
            column_config={'Exercise': st.column_config.SelectboxColumn(options=list(by_name), required=True),
                           'Sets': st.column_config.NumberColumn(min_value=1, max_value=20, step=1, required=True),
                           'Min': st.column_config.NumberColumn(min_value=1, max_value=100, step=1),
                           'Max': st.column_config.NumberColumn(min_value=1, max_value=100, step=1)})
        if st.form_submit_button('Save Template'):
            try:
                if not name.strip():
                    raise ValueError('Enter a template name.')
                prescription = []
                for row in edited.to_dict('records'):
                    if row['Exercise'] not in by_name or pd.isna(row['Sets']):
                        raise ValueError('Choose an exercise and set count for every row.')
                    if any(not pd.isna(row[col]) and int(row[col]) != row[col] for col in ('Sets', 'Min', 'Max')):
                        raise ValueError('Set counts and rep ranges must be whole numbers.')
                    prescription.append(dict(exercise_id=by_name[row['Exercise']]['id'], name=row['Exercise'],
                        set_count=int(row['Sets']), rep_min=None if pd.isna(row['Min']) else int(row['Min']),
                        rep_max=None if pd.isna(row['Max']) else int(row['Max']), rep_note='' if pd.isna(row['Notes']) else row['Notes']))
                validate_prescription(prescription)
                payload = dict(user_id=user_id, name=name.strip(), weekday=weekday, exercises=prescription)
                if selected:
                    client.table('workout_templates').update(payload).eq('user_id', user_id).eq('id', selected).execute()
                else:
                    client.table('workout_templates').insert(payload).execute()
                st.session_state['workout_notice'] = 'Template saved.'
                st.rerun()
            except Exception as exc:
                st.error(f'Template was not saved: {exc}')


def render_library(client, user_id: str, library: list[dict]) -> None:
    search = st.text_input('Search exercises')
    shown = [e for e in library if search.lower() in f"{e['name']} {e['equipment']} {e['muscle_group']}".lower()]
    st.dataframe(pd.DataFrame([dict(Exercise=e['name'], Equipment=e['equipment'], Muscle=e['muscle_group']) for e in shown]), hide_index=True, use_container_width=True)
    with st.form('custom_exercise'):
        name = st.text_input('Exercise name')
        equipment = st.selectbox('Equipment', ['Barbell', 'Dumbbells', 'Kettlebell', 'Cable', 'Machine', 'Bodyweight', 'Bike', 'Other'])
        muscle = st.text_input('Primary muscle group')
        if st.form_submit_button('Add Exercise'):
            if not name.strip():
                st.error('Enter an exercise name.')
            else:
                client.table('workout_exercises').insert(dict(user_id=user_id, name=name.strip(), equipment=equipment, muscle_group=muscle.strip())).execute()
                st.rerun()


def render_history(client, user_id: str, sessions: list[dict], library: list[dict]) -> None:
    if not sessions:
        st.info('No workouts logged yet.')
        return
    ordered = sorted(sessions, key=lambda s: (s['performed_date'], s['id']), reverse=True)
    by_id = {s['id']: s for s in ordered}
    sid = st.selectbox('Historical workout', list(by_id), format_func=lambda k: f"{by_id[k]['performed_date']} · {by_id[k]['name']} · {by_id[k]['status'].replace('_', ' ')}")
    render_logger(client, user_id, library, dict(by_id[sid], saved=True))


def render_workouts(client, user_id: str, today: date) -> None:
    st.subheader('Workouts')
    st.caption(program_label(today))
    if notice := st.session_state.pop('workout_notice', None):
        st.success(notice)
    try:
        library = rows(client, 'workout_exercises', user_id, 'name')
        templates = rows(client, 'workout_templates', user_id, 'weekday')
        sessions = rows(client, 'workout_sessions', user_id, 'performed_date')
    except Exception as exc:
        # A missing workout migration must not prevent access to nutrition pages.
        st.error('Workout storage is unavailable. Existing nutrition pages are still available.')
        with st.expander('Database setup'):
            st.write('Run migrations/001_workouts.sql in the Supabase SQL Editor.')
            st.code(str(exc))
        return
    if not templates:
        if st.button('Add Starter Program', type='primary'):
            try:
                seed_starters(client, user_id)
                st.rerun()
            except Exception as exc:
                st.error(f'Starter program was not saved: {exc}')
    # Only render the selected section: logger widgets must not appear twice.
    section = st.radio('Workout view', ['Schedule', 'Templates', 'Exercise Library', 'History'], horizontal=True, label_visibility='collapsed')
    try:
        if section == 'Schedule':
            render_schedule(client, user_id, today, templates, sessions, library)
        elif section == 'Templates':
            render_templates(client, user_id, templates, library)
        elif section == 'Exercise Library':
            render_library(client, user_id, library)
        else:
            render_history(client, user_id, sessions, library)
    except Exception as exc:
        st.error(f'Could not load workout data: {exc}')
