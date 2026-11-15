"""Analyze a folder once using real Roboflow inference; reuse the persisted results later."""
import argparse
import os
import uuid
from pathlib import Path

from app import main
from app.cache import file_digest, load_cache


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Folder containing the demo videos')
    parser.add_argument('--limit', type=int, default=20, help='Maximum videos (default: 20)')
    parser.add_argument('--force', action='store_true', help='Refresh even when cached results exist')
    args = parser.parse_args()
    if not args.folder.is_dir() or args.limit < 1:
        parser.error('Supply an existing folder and a positive --limit.')
    paths = sorted(p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower()
                   in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'})[:args.limit]
    if not paths:
        parser.error('No supported videos were found in this folder.')
    failures = 0
    for index, path in enumerate(paths, 1):
        print(f'[{index}/{len(paths)}] {path.name}', flush=True)
        if path.stat().st_size > main.MAX_UPLOAD_BYTES:
            print('  FAILED: video exceeds the dashboard upload limit.', flush=True)
            failures += 1
            continue
        digest = file_digest(path)
        if not args.force and load_cache(main.CACHE_DIR, digest, main.cache_settings()):
            print('  READY: existing results reused.', flush=True)
            continue
        video_id = str(uuid.uuid4())
        main.videos[video_id] = {'path': path.resolve(), 'filename': path.name,
                                 'size': path.stat().st_size, 'sha256': digest}
        main.jobs[video_id] = main.new_job()
        try:
            main._run_analysis(video_id)
            job = main.jobs[video_id]
            if job['status'] == 'completed' and not job.get('cache_warning'):
                print(f"  SAVED: {job['processed_frames']} frames, {len(job['events'])} events.", flush=True)
            else:
                message = job.get('error') or job.get('cache_warning') or 'Analysis failed'
                key = os.getenv('ROBOFLOW_API_KEY', '')
                if key:
                    message = message.replace(key, '[redacted]')
                print(f'  FAILED: {message}', flush=True)
                failures += 1
        finally:
            main.jobs.pop(video_id, None)
            main.videos.pop(video_id, None)
    print(f'Finished: {len(paths) - failures} ready, {failures} failed.', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(run())
