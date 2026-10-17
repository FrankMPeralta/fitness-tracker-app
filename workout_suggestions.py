"""Conservative starting-load suggestions from recorded performance only."""
from collections import Counter
from math import floor, isfinite


def number(value):
    try:
        value = float(value)
        return value if isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


def suggest_weight(item: dict, previous: list[dict], equipment: str = '', allow_increase: bool = True) -> dict:
    recorded = [s for s in previous if s.get('completed') and number(s.get('weight_lbs')) is not None]
    notes = ' '.join(str(s.get('notes') or '') for s in recorded).lower()
    equipment = 'Cable' if 'cable' in notes else equipment
    unit = 'lb per dumbbell' if 'dumbbell' in equipment.lower() else 'lb'
    step = number(item.get('weight_increment_lbs')) or 5.0
    result = dict(weight=None, unit=unit, reason='No completed weight baseline yet.', caution='')
    knee = max((number(s.get('workout_sessions', {}).get('knee_discomfort')) or 0 for s in recorded), default=0)
    knee_movement = any(word in item['name'].lower() for word in ('squat', 'lunge', 'step-up', 'step up'))
    if knee_movement and knee >= 3:
        result['caution'] = f'Previous session knee discomfort: {knee:g}/10. Hold progression; lower the load or modify the movement if uncomfortable.'

    override = number(item.get('suggested_weight_lbs'))
    if override is not None:
        return dict(result, weight=override, reason='Saved template starting weight.')
    if not recorded:
        return result
    low, high = item.get('rep_min'), item.get('rep_max')
    if low is None or high is None:
        return dict(result, weight=float(recorded[-1]['weight_lbs']), reason='Repeat the recorded load; compare duration or distance before progressing.')
    eligible = [s for s in recorded if number(s.get('reps')) is not None and float(s['reps']) >= low]
    if not eligible:
        return dict(result, reason='Previous sets were below this rep target. Establish a comfortable baseline for the new target.')
    counts = Counter(float(s['weight_lbs']) for s in eligible)
    moderate = [s for s in eligible if number(s.get('rir')) is not None and 1 <= float(s['rir']) <= 4]
    candidates = moderate or eligible
    repeated = [w for w in counts if sum(float(s['weight_lbs']) == w for s in candidates) >= 2]
    weight = max(repeated) if repeated else max(float(s['weight_lbs']) for s in candidates)
    at_weight = [s for s in eligible if float(s['weight_lbs']) == weight]
    reason = 'Repeat this tested load and build reps within the target range.'
    if weight == 0:
        return dict(result, weight=0.0, reason='Repeat bodyweight or the same assistance; build reps before adding load.')

    def reduce_load(load):
        return max(0.0, min(load - step, floor(load * 0.9 / step + 0.5) * step))

    if any(number(s.get('rir')) == 0 for s in at_weight):
        weight = reduce_load(weight)
        reason = 'A set at this load reached 0 RIR. Start lighter and retain 1-3 good reps.'
    elif len(counts) > 1 and counts[weight] == 1 and weight == max(counts):
        weight = reduce_load(weight)
        reason = 'Only one top set was tested at this load. Start slightly lighter for repeatable working sets.'
    else:
        # More reps or more prescribed sets must not trigger an automatic load increase.
        snapshots = [p for p in previous[0].get('workout_sessions', {}).get('prescription', [])
                     if p.get('exercise_id') == item.get('exercise_id')]
        unchanged = len(snapshots) == 1 and all(snapshots[0].get(k) == item.get(k) for k in ('rep_min', 'rep_max', 'set_count'))
        ready = (unchanged and len(recorded) == item['set_count'] and len(counts) == 1
                 and len(eligible) == item['set_count'] and all(float(s['reps']) >= high
                     and number(s.get('rir')) is not None and 2 <= float(s['rir']) <= 4 for s in eligible))
        if ready and allow_increase and not result['caution'] and step <= weight * 0.1:
            weight += step
            reason = 'All prescribed sets reached the top of the rep range with at least 2 RIR. Consider one small increment if technique was sound.'
        elif any(s.get('rir') is None for s in eligible):
            reason = 'Repeat the tested load; effort ratings are missing, so no increase is suggested.'
    if result['caution']:
        reason = 'Keep the load conservative while checking knee comfort.'
    elif not allow_increase:
        reason = 'Recovery week: no increase suggested. Adjust this starting load to your recovery plan.'
    return dict(result, weight=weight, reason=reason)


def weight_label(suggestion: dict) -> str:
    weight = suggestion['weight']
    if weight is None:
        return 'Baseline needed'
    if weight == 0:
        return 'Bodyweight / same assistance'
    return f"{weight:g} {suggestion['unit']}"
