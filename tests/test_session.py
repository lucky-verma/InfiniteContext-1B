import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.fetch_model import ROOT
from streaming.session import Session


class MemoryBackend:
    def __init__(self, directory, window):
        self.directory, self.window, self.cached = directory, window, []
        self.calls = 0
        self.filename = json.loads((ROOT / 'serving/model.json').read_text())['filename']

    def rpc(self, base, path, payload=None, **_):
        if path == '/v1/models':
            return {'data': [{'id': self.filename}]}
        if path == '/props':
            return {'total_slots': 1, 'cache_ram_mib': 0, 'default_generation_settings': {'n_ctx': self.window, 'params': {'n_cache_shift': 0}}}
        if path == '/slots':
            return [{'is_processing': False}]
        if path == '/tokenize':
            return {'tokens': list(payload['content'].encode())}
        if path.endswith('erase'):
            self.cached = []
        elif path.endswith('save'):
            (self.directory / payload['filename']).write_text(json.dumps(self.cached))
        elif path.endswith('restore'):
            self.cached = json.loads((self.directory / payload['filename']).read_text())
        else:
            raise AssertionError(path)
        return {}

    def stream(self, base, payload, **_):
        self.calls += 1
        prompt, count = payload['prompt'], payload['n_predict']
        shift, keep = payload['n_cache_shift'], payload['n_keep']
        if shift:
            self.cached = self.cached[:keep] + self.cached[keep + shift:]
        assert prompt[:len(self.cached)] == self.cached
        processed = len(prompt) - len(self.cached)
        self.cached = list(prompt)
        for index in range(count):
            if index:
                self.cached.append(120)
            yield {'stop': False, 'content': 'x', 'tokens': [120]}
        yield {'stop': True, 'tokens': [], 'tokens_predicted': count,
               'tokens_cached': len(self.cached), 'stop_type': 'limit',
               'timings': {'prompt_n': processed, 'cache_n': len(prompt) - processed}}


class SessionIntegrity(unittest.TestCase):
    def test_ordered_stream_replay_cancellation_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / 'kv'
            state.mkdir()
            key = root / 'key'
            key.write_text('unit-test-key')
            backend = MemoryBackend(state, 256)
            settings = {'window': 256, 'state_dir': state, 'key_file': key, 'checkpoint_interval': 4}
            with patch('streaming.session.json_request', side_effect=backend.rpc), patch('streaming.session.completion_events', side_effect=backend.stream):
                database = root / 'session.sqlite'
                with Session(database, **settings) as session:
                    for index in range(10):
                        result = list(session.append(str(index), 'a' * 40))[-1]
                        self.assertLess(result['active_tokens'], 256)
                        self.assertEqual(result['runtime_timings']['prompt_n'], 40)
                    self.assertEqual(result['total_input_tokens'], 400)
                    self.assertEqual([row['request_id'] for row in session.search('a*', limit=2)], ['9', '8'])
                    with self.assertRaises(ValueError):
                        session.search('"')
                    calls = backend.calls
                    self.assertTrue(list(session.append('9', 'a' * 40))[-1]['replayed'])
                    self.assertEqual(backend.calls, calls)
                    with self.assertRaises(ValueError):
                        list(session.append('9', 'changed'))
                    with self.assertRaises(RuntimeError):
                        Session(database, **settings)
                    with self.assertRaises(ValueError):
                        list(session.append('oversize', 'a' * 1000))
                    stream = session.append('interrupted', 'b' * 10, generate=3)
                    self.assertEqual(next(stream)['type'], 'accepted')
                    self.assertEqual(next(stream)['type'], 'delta')
                    with self.assertRaises(RuntimeError):
                        list(session.append('parallel', 'c'))
                    stream.close()
                    self.assertEqual(session.search('b*'), [])
                    with self.assertRaises(RuntimeError):
                        list(session.append('later', 'c'))
                    retried = list(session.append('interrupted', 'b' * 10, generate=3))[-1]
                    self.assertEqual(retried['text'], 'xxx')
                    self.assertEqual(retried['generated_tokens'], 3)
                    self.assertEqual(session.search('b*')[0]['request_id'], 'interrupted')
                backend.cached = [999]
                with Session(database, **settings) as session:
                    replayed = list(session.append('interrupted', 'b' * 10, generate=3))[-1]
                    self.assertTrue(replayed['replayed'])
                    follow = list(session.append('next', 'd' * 10))[-1]
                    self.assertEqual(follow['total_input_tokens'], 420)
                    self.assertEqual(len(session.history()), 12)
                    self.assertLessEqual(len(list(state.glob(session.state['session_id'] + '-*.bin'))), 2)
                    snapshot = session.snapshot_path(session.state['snapshot']['file'])
                snapshot.write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    Session(database, **settings)


if __name__ == '__main__':
    unittest.main()
