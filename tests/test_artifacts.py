import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.fetch_model import verify


class ArtifactIntegrity(unittest.TestCase):
    def test_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.gguf'
            payload = b'test artifact bytes'
            path.write_bytes(payload)
            manifest = {'size_bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
            verify(path, manifest)
            path.write_bytes(b'X' + payload[1:])
            with self.assertRaises(ValueError):
                verify(path, manifest)
            self.assertEqual(path.read_bytes(), b'X' + payload[1:])


if __name__ == '__main__':
    unittest.main()
