"""In-memory stand-in for page tests; never connects to the live account."""
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4


class FakeClient:
    def __init__(self):
        self.tables = {name: [] for name in ('workout_exercises', 'workout_templates', 'workout_plans', 'workout_sessions', 'workout_sets')}

    def table(self, name):
        return Query(self, name)

    def rpc(self, name, args):
        assert name == 'save_workout'
        session = dict(args['p_session'], user_id='user-a')
        sid = session['id']
        self.tables['workout_sessions'] = [s for s in self.tables['workout_sessions'] if s['id'] != sid] + [session]
        self.tables['workout_sets'] = [s for s in self.tables['workout_sets'] if s['session_id'] != sid] + [dict(s, id=str(uuid4()), session_id=sid, user_id='user-a') for s in args['p_sets']]
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=sid))


class Query:
    def __init__(self, client, table):
        self.client, self.name = client, table
        self.filters, self.orders = [], []
        self.operation, self.payload = 'select', None
        self.start, self.end = 0, 100000
        self.join = False

    def select(self, columns):
        self.join = '!inner' in columns
        return self

    def eq(self, key, value):
        self.filters.append(lambda row: row.get(key) == value)
        return self

    def neq(self, key, value):
        self.filters.append(lambda row: row.get(key) != value)
        return self

    def gte(self, key, value):
        self.filters.append(lambda row: row.get(key, '') >= value)
        return self

    def lte(self, key, value):
        self.filters.append(lambda row: row.get(key, '') <= value)
        return self

    def order(self, key, **kwargs):
        self.orders.append((key, kwargs.get('desc', False)))
        return self

    def range(self, start, end):
        self.start, self.end = start, end
        return self

    def limit(self, count):
        self.end = count - 1
        return self

    def upsert(self, payload, on_conflict='', ignore_duplicates=False):
        self.operation, self.payload = 'upsert', payload
        self.conflict, self.ignore = on_conflict.split(','), ignore_duplicates
        return self

    def insert(self, payload):
        self.operation, self.payload = 'insert', payload
        return self

    def update(self, payload):
        self.operation, self.payload = 'update', payload
        return self

    def execute(self):
        table = self.client.tables[self.name]
        if self.join:
            joined = []
            for row in table:
                parent = next((s for s in self.client.tables['workout_sessions'] if s['id'] == row['session_id']), None)
                if parent:
                    joined.append(dict(row, workout_sessions=parent, **{
                        f'workout_sessions.{k}': v for k, v in parent.items()}))
            table = joined
        selected = [row for row in table if all(f(row) for f in self.filters)]
        if self.operation in ('insert', 'upsert'):
            for payload in self.payload if isinstance(self.payload, list) else [self.payload]:
                old = next((r for r in table if all(r.get(k) == payload.get(k) for k in self.conflict)), None) if self.operation == 'upsert' else None
                if old is None:
                    table.append(dict(deepcopy(payload), id=payload.get('id', str(uuid4()))))
                elif not self.ignore:
                    old.update(deepcopy(payload))
        elif self.operation == 'update':
            for row in selected:
                row.update(deepcopy(self.payload))
        else:
            for key, descending in reversed(self.orders):
                selected.sort(key=lambda r: r['workout_sessions']['performed_date'] if key == 'workout_sessions(performed_date)' else r[key], reverse=descending)
        return SimpleNamespace(data=deepcopy(selected[self.start:self.end + 1]))
