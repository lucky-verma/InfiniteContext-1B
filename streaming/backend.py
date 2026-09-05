"""Small authenticated client for the native runtime's token and state APIs."""

import json
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def request(base, path, payload=None, *, key='', timeout=120):
    parsed = urlparse(base)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('backend must be an HTTP(S) URL without embedded credentials')
    if parsed.scheme == 'http' and parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('remote backends require HTTPS')
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    body = None if payload is None else json.dumps(payload).encode()
    try:
        return urlopen(Request(base.rstrip('/') + path, data=body, headers=headers), timeout=timeout)
    except HTTPError as error:
        detail = error.read(2048).decode(errors='replace')
        if key:
            detail = detail.replace(key, '<redacted>')
        raise RuntimeError(f'backend HTTP {error.code}: {detail}') from None


def json_request(base, path, payload=None, *, key=''):
    with request(base, path, payload, key=key) as response:
        return json.load(response)


def completion_events(base, payload, *, key=''):
    with request(base, '/completion', payload, key=key) as response:
        for line in response:
            if line.startswith(b'data: '):
                event = json.loads(line[6:])
                if 'error' in event:
                    raise RuntimeError(str(event['error']))
                yield event
