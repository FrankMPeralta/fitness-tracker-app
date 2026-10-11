-- Additive migration: existing nutrition tables and rows are untouched.
begin;

create table if not exists public.workout_exercises (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null check (length(trim(name)) > 0),
  equipment text not null default '',
  muscle_group text not null default '',
  unique (user_id, name), unique (user_id, id)
);

create table if not exists public.workout_templates (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null check (length(trim(name)) > 0),
  weekday integer not null check (weekday between 0 and 6),
  exercises jsonb not null default '[]' check (jsonb_typeof(exercises) = 'array'),
  unique (user_id, name), unique (user_id, id)
);

create table if not exists public.workout_plans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  template_id uuid not null,
  scheduled_date date not null,
  schedule_key text not null,
  unique (user_id, schedule_key), unique (user_id, id),
  foreign key (user_id, template_id) references public.workout_templates(user_id, id)
);

create table if not exists public.workout_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  planned_id uuid,
  performed_date date not null,
  name text not null check (length(trim(name)) > 0),
  prescription jsonb not null check (jsonb_typeof(prescription) = 'array'),
  status text not null check (status in ('in_progress', 'completed')),
  knee_discomfort integer check (knee_discomfort between 0 and 10),
  notes text not null default '',
  unique (user_id, id), unique (user_id, planned_id),
  foreign key (user_id, planned_id) references public.workout_plans(user_id, id)
);

create table if not exists public.workout_sets (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid not null,
  exercise_id uuid not null,
  exercise_position integer not null check (exercise_position >= 0),
  set_number integer not null check (set_number between 1 and 20),
  weight_lbs numeric check (weight_lbs >= 0),
  reps integer check (reps >= 0),
  rir numeric check (rir between 0 and 10),
  completed boolean not null default false,
  notes text not null default '',
  check (not completed or reps > 0),
  unique (session_id, exercise_position, set_number),
  foreign key (user_id, session_id) references public.workout_sessions(user_id, id) on delete cascade,
  foreign key (user_id, exercise_id) references public.workout_exercises(user_id, id)
);

create index if not exists workout_plans_calendar on public.workout_plans(user_id, scheduled_date);
create index if not exists workout_sessions_history on public.workout_sessions(user_id, performed_date);
create index if not exists workout_sets_history on public.workout_sets(user_id, exercise_id);

alter table public.workout_exercises enable row level security;
alter table public.workout_templates enable row level security;
alter table public.workout_plans enable row level security;
alter table public.workout_sessions enable row level security;
alter table public.workout_sets enable row level security;

do $$
declare t text;
begin
  foreach t in array array['workout_exercises', 'workout_templates', 'workout_plans', 'workout_sessions', 'workout_sets'] loop
    execute format('drop policy if exists user_isolation on public.%I', t);
    execute format('create policy user_isolation on public.%I for all to authenticated using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id)', t);
    execute format('grant select, insert, update, delete on public.%I to authenticated', t);
    execute format('revoke all on public.%I from anon', t);
  end loop;
end $$;

-- One transaction saves a session and all its sets; failures retain the old log.
-- SECURITY INVOKER keeps the caller's RLS policies in force.
create or replace function public.save_workout(p_session jsonb, p_sets jsonb)
returns uuid language plpgsql security invoker set search_path = public
as $$
declare
  owner_id uuid := auth.uid();
  saved_session_id uuid := (p_session->>'id')::uuid;
begin
  if owner_id is null or saved_session_id is null then
    raise exception 'Authenticated user and session ID required';
  end if;
  if jsonb_typeof(p_sets) <> 'array' then
    raise exception 'Sets must be an array';
  end if;
  if exists (
    select 1 from jsonb_array_elements(p_session->'prescription') item
    where not exists (select 1 from workout_exercises e
      where e.id = (item->>'exercise_id')::uuid and e.user_id = owner_id)
  ) then
    raise exception 'Exercise does not belong to this user';
  end if;
  insert into workout_sessions(id, user_id, planned_id, performed_date, name, prescription, status, knee_discomfort, notes)
  values (saved_session_id, owner_id, nullif(p_session->>'planned_id', '')::uuid,
    (p_session->>'performed_date')::date, p_session->>'name', p_session->'prescription',
    p_session->>'status', (p_session->>'knee_discomfort')::integer, coalesce(p_session->>'notes', ''))
  on conflict (id) do update set
    performed_date = excluded.performed_date, name = excluded.name,
    prescription = excluded.prescription, status = excluded.status,
    knee_discomfort = excluded.knee_discomfort, notes = excluded.notes;

  delete from workout_sets where session_id = saved_session_id and user_id = owner_id;
  insert into workout_sets(user_id, session_id, exercise_id, exercise_position, set_number, weight_lbs, reps, rir, completed, notes)
  select owner_id, saved_session_id, x.exercise_id, x.exercise_position, x.set_number,
    x.weight_lbs, x.reps, x.rir, x.completed, coalesce(x.notes, '')
  from jsonb_to_recordset(p_sets) as x(exercise_id uuid, exercise_position integer,
    set_number integer, weight_lbs numeric, reps integer, rir numeric, completed boolean, notes text);
  return saved_session_id;
end $$;

revoke all on function public.save_workout(jsonb, jsonb) from public, anon;
grant execute on function public.save_workout(jsonb, jsonb) to authenticated;
commit;
