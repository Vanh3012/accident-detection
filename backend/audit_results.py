"""Audit genuine cached predictions. This reports model output, not accuracy against ground truth."""
import argparse
import json
from collections import Counter
from pathlib import Path

from app import main
from app.cache import file_digest


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    paths = sorted(p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower()
                   in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'})
    rows, missing, labels = [], [], Counter()
    for path in paths:
        digest = file_digest(path)
        job = main.cached_analysis(digest)
        if job is None:
            missing.append(path.name)
            continue
        peaks = {}
        uncertain = 0
        for frame in job['results']:
            scores = {}
            for prediction in frame['predictions']:
                label = prediction['class']
                labels[label] += 1
                if label in main.ACCIDENT_TYPES:
                    category = main.ACCIDENT_TYPES[label][0]
                    scores[category] = max(scores.get(category, 0), prediction['confidence'])
                    peaks[category] = max(peaks.get(category, 0), prediction['confidence'])
            ranked = sorted(scores.values(), reverse=True)
            if len(ranked) > 1 and ranked[0] >= main.THRESHOLD and ranked[0] - ranked[1] < .1:
                uncertain += 1
        rows.append({'filename': path.name, 'sha256': digest, 'frames': len(job['results']),
                     'events': job['events'], 'cached_at': job['cached_at'],
                     'peak_confidence_by_type': peaks, 'ambiguous_frames': uncertain,
                     'accident_frames_by_threshold': {
                         str(t): sum(main.classify(f['predictions'], t)['status'] == 'accident'
                                     for f in job['results']) for t in [.3, .5, .7, .9]}})
    report = {'settings': main.cache_settings(), 'ready': len(rows), 'missing': missing,
              'prediction_labels': dict(labels), 'videos': rows,
              'accuracy': None, 'note': 'No ground truth supplied; accuracy and optimal threshold are not measured. Threshold counts use returned API predictions only; detections filtered by the inference server cannot be recovered.'}
    main.CACHE_DIR.mkdir(exist_ok=True)
    (main.CACHE_DIR / 'demo_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    annotations = main.ROOT / 'ground_truth.template.json'
    if not annotations.exists():
        annotations.write_text(json.dumps({'class_ids': {'0': 'normal', '1': 'vehicle_vehicle',
            '2': 'motorcycle', '3': 'single_vehicle', '4': 'pedestrian'},
            'videos': [{'filename': p.name, 'incidents': None} for p in paths]}, indent=2), encoding='utf-8')
    print(json.dumps({'ready': len(rows), 'missing': missing,
                     'frames': sum(r['frames'] for r in rows),
                     'events': sum(len(r['events']) for r in rows),
                     'without_events': [r['filename'] for r in rows if not r['events']],
                     'ambiguous_frames': sum(r['ambiguous_frames'] for r in rows)}, ensure_ascii=False), flush=True)
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(run())
