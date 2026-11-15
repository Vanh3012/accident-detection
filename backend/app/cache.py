"""Persistent cache of genuine, completed inference jobs, keyed by video bytes and settings."""
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(video_sha256: str, settings: dict) -> str:
    data = json.dumps({'video': video_sha256, 'settings': settings, 'schema': 1}, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


def load_cache(directory: Path, video_sha256: str, settings: dict):
    try:
        payload = json.loads((directory / f'{cache_key(video_sha256, settings)}.json').read_text(encoding='utf-8'))
        if (payload['video_sha256'] != video_sha256 or payload['settings'] != settings
                or payload['job']['status'] != 'completed' or not payload['job']['results']):
            return None
        return {**payload['job'], 'source': 'cache', 'cached_at': payload['cached_at']}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_cache(directory: Path, video_sha256: str, settings: dict, job: dict) -> str:
    if job['status'] != 'completed' or not job['results']:
        raise ValueError('Only completed, nonempty inference jobs can be cached.')
    directory.mkdir(parents=True, exist_ok=True)
    cached_at = datetime.now(timezone.utc).isoformat()
    target = directory / f'{cache_key(video_sha256, settings)}.json'
    temporary = directory / f'.{uuid.uuid4().hex}.tmp'
    try:
        temporary.write_text(json.dumps({'video_sha256': video_sha256, 'settings': settings,
                                         'cached_at': cached_at, 'job': job}, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return cached_at
