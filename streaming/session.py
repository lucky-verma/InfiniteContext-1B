"""Durable, bounded-token sessions for one leased native backend slot."""

import fcntl
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

from scripts.fetch_model import ROOT
from streaming.backend import completion_events, json_request


class Session:
    def __init__(self, database, *, endpoint='http://127.0.0.1:18081', window=4096,
                 anchors=4, checkpoint_interval=16, key_file=None, state_dir=None, ignore_eos=False):
        if type(window) is not int or type(anchors) is not int or type(checkpoint_interval) is not int or window < 256 or not 0 <= anchors < window - 3 or checkpoint_interval < 1:
            raise ValueError('invalid window, anchor count, or checkpoint interval')
        self.endpoint, self.window, self.anchors = endpoint, window, anchors
        if type(ignore_eos) is not bool:
            raise ValueError('ignore_eos must be boolean')
        self.ignore_eos = ignore_eos
        self.checkpoint_interval = checkpoint_interval
        self.state_dir = Path(state_dir or ROOT / '.sessions/kv').resolve()
        self.state_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        key_file = Path(key_file or ROOT / '.sessions/backend.key')
        self.key = key_file.read_text().strip()
        if not self.key:
            raise ValueError('backend key is empty')
        lease_name = hashlib.sha256(self.key.encode()).hexdigest()[:24] + '.lock'
        database = Path(database).resolve()
        database.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.leases = []
        try:
            for path in (database.with_suffix(database.suffix + '.lock'), self.state_dir / lease_name):
                lease = path.open('a')
                self.leases.append(lease)
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            for lease in self.leases:
                lease.close()
            if isinstance(error, BlockingIOError):
                raise RuntimeError('database or backend slot is already leased by another session') from None
            raise
        self.db = None
        try:
            model = json.loads((ROOT / 'serving/model.json').read_text())
            runtime = json.loads((ROOT / 'serving/runtime.json').read_text())
            model_ids = {item['id'] for item in self.rpc('/v1/models')['data']}
            if model['filename'] not in model_ids:
                raise ValueError('backend model identity does not match the configured model')
            props = self.rpc('/props')
            actual_window = props['default_generation_settings']['n_ctx']
            if 'n_cache_shift' not in props['default_generation_settings']['params']:
                raise ValueError('backend lacks the position-aware streaming protocol; rebuild the pinned runtime')
            if props['total_slots'] != 1 or actual_window != window:
                raise ValueError(f'backend must have one slot with context {window}; reported {actual_window}')
            if props.get('cache_ram_mib') != 0:
                raise ValueError('stateful streaming requires the shared prompt cache to be disabled (--cache-ram 0)')
            config = {'protocol': 1, 'window': window, 'anchors': anchors, 'model_sha256': model['sha256'],
                      'runtime_revision': runtime['revision'], 'patch_sha256': runtime['patch_sha256']}
            if ignore_eos:
                config['ignore_eos'] = True
            descriptor = os.open(database, os.O_RDWR | os.O_CREAT, 0o600)
            os.close(descriptor)
            self.db = sqlite3.connect(database)
            tables = {row[0] for row in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables and not {'state', 'requests'} <= tables:
                raise ValueError('database is not an InfiniteContext session store')
            if 'state' in tables:
                prior = self.db.execute('SELECT data FROM state WHERE id=1').fetchone()
                if prior and json.loads(prior[0])['config'] != config:
                    raise ValueError('saved session model/runtime/window configuration differs')
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK (id=1), data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS requests (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT UNIQUE NOT NULL,
                    text TEXT NOT NULL, generate INTEGER NOT NULL, input_tokens TEXT NOT NULL,
                    status TEXT NOT NULL, result TEXT, active_tokens TEXT);
            ''')
            if 'history_search' not in tables:
                self.db.executescript('''
                    BEGIN IMMEDIATE;
                    CREATE VIRTUAL TABLE history_search USING fts5(text, content='requests', content_rowid='seq');
                    CREATE TRIGGER history_insert AFTER INSERT ON requests BEGIN
                        INSERT INTO history_search(rowid,text) VALUES (new.seq,new.text);
                    END;
                    INSERT INTO history_search(history_search) VALUES ('rebuild');
                    COMMIT;
                ''')
            row = self.db.execute('SELECT data FROM state WHERE id=1').fetchone()
            self.state = json.loads(row[0]) if row else {'config': config, 'session_id': uuid.uuid4().hex,
                'active': [], 'last_seq': 0, 'input_tokens': 0, 'output_tokens': 0,
                'snapshot': None, 'previous_snapshot': None}
            if self.state['config'] != config:
                raise ValueError('saved session model/runtime/window configuration differs')
            if uuid.UUID(hex=self.state['session_id']).hex != self.state['session_id']:
                raise ValueError('invalid session identity')
            if not row:
                with self.db:
                    self.db.execute('INSERT INTO state VALUES (1, ?)', (json.dumps(self.state),))
            self.needs_restore = True
            self.in_flight = False
            self.current_events = None
            self.restore()
            self.prune_snapshots()
        except BaseException:
            self.close()
            raise

    def rpc(self, path, payload=None):
        try:
            return json_request(self.endpoint, path, payload, key=self.key)
        except (OSError, ValueError, RuntimeError):
            if hasattr(self, 'state'):
                self.needs_restore = True
            raise

    def payload(self, tokens, generate, shift=0):
        return {'id_slot': 0, 'prompt': tokens, 'n_predict': generate, 'temperature': 0,
                'seed': 42, 'stream': True, 'return_tokens': True, 'cache_prompt': True,
                'n_cache_reuse': 0, 'n_cache_shift': shift, 'n_keep': self.anchors, 'stop': [],
                'ignore_eos': self.ignore_eos}

    def snapshot_path(self, name):
        if Path(name).name != name or not name.startswith(self.state['session_id'] + '-') or not name.endswith('.bin'):
            raise ValueError('invalid session snapshot filename')
        return self.state_dir / name

    def restore(self):
        deadline = time.monotonic() + 30
        while any(slot['is_processing'] for slot in self.rpc('/slots')):
            if time.monotonic() >= deadline:
                raise TimeoutError('backend did not become idle before session restore')
            time.sleep(0.05)
        snapshot = self.state['snapshot']
        if snapshot:
            path = self.snapshot_path(snapshot['file'])
            with path.open('rb') as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != snapshot['sha256']:
                    raise ValueError('session snapshot integrity check failed')
            self.rpc('/slots/0?action=restore', {'filename': snapshot['file']})
            start = snapshot['seq']
        else:
            self.rpc('/slots/0?action=erase', {})
            start = 0
        rows = self.db.execute("SELECT active_tokens,result FROM requests WHERE status='complete' AND seq>? ORDER BY seq", (start,))
        for encoded, encoded_result in rows:
            tokens, result = json.loads(encoded), json.loads(encoded_result)
            # Restore the consumed prefix; a final sampled token can still be pending.
            tokens = tokens[:result['cached_tokens']]
            final = None
            with closing(completion_events(self.endpoint, self.payload(tokens, 0, result['evicted_tokens']), key=self.key)) as events:
                for event in events:
                    if event.get('stop'):
                        final = event
            if final is None:
                raise RuntimeError('session replay ended without a completion event')
        self.needs_restore = False

    def save_snapshot(self, seq):
        name = self.state['session_id'] + '-' + uuid.uuid4().hex + '.bin'
        self.rpc('/slots/0?action=save', {'filename': name})
        path = self.snapshot_path(name)
        with path.open('r+b') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
            os.fsync(f.fileno())
        directory = os.open(self.state_dir, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return {'file': name, 'sha256': digest, 'seq': seq, 'size_bytes': path.stat().st_size}

    def prune_snapshots(self):
        retained = {s['file'] for s in (self.state['snapshot'], self.state['previous_snapshot']) if s}
        for path in self.state_dir.glob(self.state['session_id'] + '-*.bin'):
            if path.name not in retained:
                self.snapshot_path(path.name).unlink()

    def history(self, *, after=0, limit=100):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('history requires a nonnegative offset and limit 1-1000')
        rows = self.db.execute('SELECT seq,request_id,text,status,result FROM requests WHERE seq>? ORDER BY seq LIMIT ?', (after, limit))
        return [{'sequence': seq, 'request_id': rid, 'text': text, 'status': status,
                 'result': json.loads(result) if result else None} for seq, rid, text, status, result in rows]

    def search(self, query, *, limit=5):
        """Retrieve committed source records; callers budget and cite any reinjection.

        This is lexical FTS5 retrieval, not guaranteed semantic or neural recall.
        Newer matching records rank first so updates remain visible.
        """
        if not isinstance(query, str) or not 1 <= len(query.encode()) <= 1024 or type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError('search requires a bounded FTS5 query and limit 1..20')
        try:
            rows = self.db.execute('''SELECT r.seq,r.request_id,r.text FROM history_search
                JOIN requests r ON r.seq=history_search.rowid
                WHERE history_search MATCH ? AND r.status='complete'
                ORDER BY r.seq DESC LIMIT ?''', (query, limit))
            return [{'sequence': seq, 'request_id': rid, 'text': text} for seq, rid, text in rows]
        except sqlite3.OperationalError as error:
            raise ValueError('invalid FTS5 query') from error

    def append(self, request_id, text, *, generate=0):
        if self.in_flight:
            raise RuntimeError('finish or close the active append before starting another')
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128 or not isinstance(text, str):
            raise ValueError('request_id and text must be strings, with a nonempty bounded ID')
        if len(text.encode()) > 65536 or type(generate) is not int or not 0 <= generate < self.window - self.anchors - 1:
            raise ValueError('request text or generation count exceeds the configured limit')
        existing = self.db.execute('SELECT seq,text,generate,status,result,input_tokens FROM requests WHERE request_id=?', (request_id,)).fetchone()
        if existing:
            seq, original, original_generate, status, encoded_result, encoded_input = existing
            if original != text or original_generate != generate:
                raise ValueError('request ID was already used with different content or generation settings')
            if status == 'complete':
                yield {'type': 'complete', 'request_id': request_id, 'replayed': True, **json.loads(encoded_result)}
                return
            incoming = json.loads(encoded_input)
        else:
            pending = self.db.execute("SELECT request_id FROM requests WHERE status!='complete' ORDER BY seq LIMIT 1").fetchone()
            if pending:
                raise RuntimeError(f'retry the unfinished request before appending: {pending[0]}')
            incoming = self.rpc('/tokenize', {'content': text, 'add_special': False, 'parse_special': True})['tokens']
            budget = self.window - generate - 1
            if len(incoming) > budget - self.anchors - 2:
                raise ValueError('input chunk is too large; split it into smaller segments before appending')
            if not incoming and (not self.state['active'] or generate == 0):
                raise ValueError('empty input requires an existing session and positive generation count')
            with self.db:
                cursor = self.db.execute("INSERT INTO requests(request_id,text,generate,input_tokens,status) VALUES (?,?,?,?,'pending')",
                    (request_id, text, generate, json.dumps(incoming)))
                seq = cursor.lastrowid
        if self.needs_restore:
            self.restore()
        combined = self.state['active'] + incoming
        budget = self.window - generate - 1
        dropped = max(0, len(combined) - budget)
        active = combined[:self.anchors] + combined[self.anchors + dropped:] if dropped else combined
        if not active or len(active) >= self.window:
            raise ValueError('invalid active token budget')
        committed = False
        self.in_flight = True
        started = time.monotonic()
        first_token = None
        content, partial_tokens, final = [], [], None
        try:
            yield {'type': 'accepted', 'request_id': request_id, 'sequence': seq, 'restart': existing is not None}
            self.current_events = completion_events(self.endpoint, self.payload(active, generate, dropped), key=self.key)
            for event in self.current_events:
                if event.get('stop'):
                    final = event
                else:
                    piece = event.get('content', '')
                    content.append(piece)
                    partial_tokens.extend(event.get('tokens', []))
                    if piece and first_token is None:
                        first_token = time.monotonic() - started
                    if piece:
                        yield {'type': 'delta', 'request_id': request_id, 'text': piece}
            if final is None:
                raise RuntimeError('stream ended without a completion event')
            generated = final.get('tokens') or partial_tokens
            unreturned = final['tokens_predicted'] - len(generated)
            if len(generated) > generate or not (unreturned == 0 or unreturned == 1 and final['stop_type'] == 'eos'):
                raise RuntimeError('runtime token accounting does not match the completion')
            after = active + generated
            if len(after) >= self.window:
                raise RuntimeError('runtime exceeded the active window budget')
            cached = final['tokens_cached']
            if type(cached) is not int or not 0 <= len(after) - cached <= 1:
                raise RuntimeError('runtime cache does not match the committed token prefix')
            updated = dict(self.state, active=after, last_seq=seq,
                input_tokens=self.state['input_tokens'] + len(incoming),
                output_tokens=self.state['output_tokens'] + len(generated))
            checkpoint_seconds = 0.0
            if not self.state['snapshot'] or seq % self.checkpoint_interval == 0:
                checkpoint_started = time.monotonic()
                snapshot = self.save_snapshot(seq)
                checkpoint_seconds = time.monotonic() - checkpoint_started
                updated.update(snapshot=snapshot, previous_snapshot=self.state['snapshot'])
            result = {'sequence': seq, 'text': ''.join(content), 'input_tokens': len(incoming),
                'generated_tokens': len(generated), 'active_tokens': len(after), 'evicted_tokens': dropped,
                'cached_tokens': cached, 'stop_type': final['stop_type'],
                'runtime_generated_steps': final['tokens_predicted'],
                'total_input_tokens': updated['input_tokens'], 'total_output_tokens': updated['output_tokens'],
                'ttft_seconds': first_token, 'runtime_timings': final.get('timings', {}),
                'checkpoint_seconds': checkpoint_seconds, 'snapshot_bytes': updated['snapshot']['size_bytes']}
            with self.db:
                self.db.execute("UPDATE requests SET status='complete',result=?,active_tokens=? WHERE request_id=?",
                    (json.dumps(result), json.dumps(after), request_id))
                self.db.execute('UPDATE state SET data=? WHERE id=1', (json.dumps(updated),))
            self.state, committed = updated, True
            self.prune_snapshots()
            yield {'type': 'complete', 'request_id': request_id, 'replayed': False, **result}
        finally:
            if self.current_events is not None:
                self.current_events.close()
                self.current_events = None
            self.in_flight = False
            if not committed:
                self.needs_restore = True
                if self.db is not None:
                    with self.db:
                        self.db.execute("UPDATE requests SET status='failed' WHERE request_id=?", (request_id,))

    def close(self):
        if getattr(self, 'current_events', None) is not None:
            self.current_events.close()
            self.current_events = None
        if self.db is not None:
            self.db.close()
            self.db = None
        for lease in self.leases:
            if not lease.closed:
                lease.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
