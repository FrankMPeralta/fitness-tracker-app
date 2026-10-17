# Workout expansion

## Architecture and scope

The existing `app.py` still owns authentication, Mountain Time date defaults,
nutrition, body metrics, charts, and exports. It passes the authenticated Supabase
client, user ID, and local date to `workouts.render_workouts` through a new
Workouts navigation entry. No existing nutrition tables are modified.

The first increment implements:

- Editable templates and an exercise library, including custom exercises.
- The four proposed strength templates and optional conditioning template.
- A weekly schedule, manual additions, and moving scheduled workouts.
- Workout logging with blank starting weights, optional RIR, per-set notes and
  completion, substitutions, workout recovery notes, and knee discomfort.
- Previous completed exercise performance and editable workout history.
- Suggested starting weights from the latest completed exercise session, with an
  explanation, optional blank-field filling, and saved template overrides.
- Draft/completed session states, stored in Supabase after Save Workout.
- Mountain Time dates and the 16-week program labels, including recovery weeks
  4, 8, 12 and a week-16 assessment label. Labels do not alter lifting targets.

Weights are pounds as entered. For dumbbells use the same convention each time;
for assisted exercises record assistance in notes. Timed work can have blank
reps and weight, with duration/distance in notes. A completed workout can contain
skipped sets; individual completion flags preserve what was actually performed.

## Tables and relationships

| Table | Purpose |
| --- | --- |
| `workout_exercises` | User-owned exercise names, equipment, muscle group |
| `workout_templates` | User-owned name, usual weekday, ordered JSON exercise prescriptions |
| `workout_plans` | Date and template reference; separate from actual training |
| `workout_sessions` | Optional plan reference, actual date, prescription snapshot, status and recovery notes |
| `workout_sets` | Session and exercise references, position, set number, actual weight/reps/RIR and notes |

Every table uses `user_id` and authenticated-user RLS. Composite foreign keys
prevent linking a user's sets or plans to another user's records. Templates use
an ordered JSON prescription for a compact editor; each entry carries an
exercise ID, name, set count and rep range. Sessions snapshot that prescription
so changing a template cannot rewrite history. The `save_workout` SQL function
runs as the caller, under RLS, and replaces session/set data atomically.

A weekly schedule key makes repeated Schedule This Week clicks harmless, even
after a workout has moved. A plan can have at most one saved session. New logs
retain a stable ID through reruns, and repeated saves update that session.

## Activation

1. Run `migrations/001_workouts.sql` in the existing project's Supabase SQL Editor.
   It is transactional and can be rerun. It adds workout tables only.
2. Deploy `app.py` and `workouts.py` together. The existing requirements suffice.
3. Open Workouts, select Add Starter Program, then Schedule This Week.
4. Open a workout and save it. Reopen it from Schedule or History to continue or
   edit the recorded sets.

If the migration has not been applied, the Workouts page reports storage as
unavailable while nutrition pages remain accessible. Browser form entries must
be saved before closing the page; Save Workout with completion unchecked stores
an in-progress session that can be resumed on another device.

The local code and live Supabase schema were inspected. The live project contains
the four expected nutrition tables, each with a user ID and RLS enabled. The
workout migration was applied successfully to the existing Supabase project on
October 11, 2026. Run it separately when setting up another database project.

## Suggested weights

Suggested weights use completed sets and recorded RIR, without estimating a
one-rep maximum. Repeated challenging loads are preferred over easy ramp-up sets.
A single heavier top set or a set at zero RIR yields a more conservative starting
load. Increases require all prescribed sets at the rep ceiling, 2-4 RIR, unchanged
rep/set targets, and an available increment no greater than 10 percent. Missing
RIR, incomplete sets, and recovery weeks do not trigger increases. Squat/lunge/
step-up suggestions flag a previous session knee score of 3 or higher and block
automatic progression; this session-level flag does not identify a pain trigger.

These are adjustable heuristics, not measured capacity or a technique assessment.
Bodyweight/assistance and timed work are held steady. Unlogged exercises get no
invented starting weight. Dumbbell loads are per hand; cable notes retain cable
units. Use the same exercise variation and machine as the reference session.

Each exercise shows the suggestion and its dated reference in the workout logger.
Use Suggested Weights fills only blank, uncompleted weight fields; it does not
save a workout or fill repetitions or completion flags. Templates offers optional
Start lb and Step lb columns. Clearing Start lb restores automatic suggestions.
No additional tables or migrations are needed.

## Next increments

1. **Progress monitoring:** metrics from completed sets, comparable-exercise
   charts, volume trends, records, and four-week summaries. Extend the current
   suggestions with technique ratings; keep progression under user control.
2. **Integration:** combine weekly nutrition/body-weight aggregates with training
   performance, then add editable phase dates and saved phase-specific nutrition
   targets without replacing the existing active macro goal.
3. **Recovery and conditioning:** structured cardio duration and modality,
   exercise-specific discomfort trends, and consistency analytics.

No calorie prescriptions, medical recommendations, or automatic weight increases
are part of this increment. The program's dates and proposed exercises are from
the supplied project brief.

## Verification

Run the focused Python checks with the project virtual environment:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The Streamlit tests use an in-memory Supabase stand-in, so they never write live
fitness data. `tests/check_database.mjs` verifies migration execution/rerun,
nutrition preservation, cross-user RLS/foreign-key failures, anonymous denial,
and rollback when a set save fails in temporary PostgreSQL (PGlite). Install
PGlite separately and point `PGLITE_PATH` at its `dist/index.js` to run it.
`tests/check_layout.mjs` checks desktop and iPhone layouts using Playwright against
the synthetic preview in `tests/preview_workouts.py`; neither uses live records.
