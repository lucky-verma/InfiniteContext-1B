"""Process ordered JSONL input with durable IDs and streaming output events."""

import argparse
import json
from pathlib import Path
import sys

from scripts.fetch_model import ROOT
from streaming.session import Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=ROOT / '.sessions/default.sqlite')
    parser.add_argument('--endpoint', default='http://127.0.0.1:18081')
    parser.add_argument('--window', type=int, default=4096)
    parser.add_argument('--anchors', type=int, default=4)
    parser.add_argument('--checkpoint-interval', type=int, default=16)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--history-after', type=int)
    parser.add_argument('--search', help='FTS5 query over committed input records; does not generate or reinject text')
    args = parser.parse_args()
    with Session(args.database, endpoint=args.endpoint, window=args.window, anchors=args.anchors,
                 checkpoint_interval=args.checkpoint_interval, key_file=args.key_file) as session:
        if args.search is not None:
            for item in session.search(args.search):
                print(json.dumps(item, ensure_ascii=False))
            return
        if args.history_after is not None:
            for item in session.history(after=args.history_after):
                print(json.dumps(item, ensure_ascii=False))
            return
        for line in sys.stdin:
            item = json.loads(line)
            if not isinstance(item, dict) or set(item) - {'id', 'text', 'generate'} or not {'id', 'text'} <= set(item):
                raise ValueError('each JSONL record requires id and text; generate is optional')
            stream = session.append(item['id'], item['text'], generate=item.get('generate', 0))
            try:
                for event in stream:
                    print(json.dumps(event, ensure_ascii=False), flush=True)
            finally:
                stream.close()


if __name__ == '__main__':
    main()
