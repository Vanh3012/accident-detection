from fastapi import BackgroundTasks

from app import main
from app.cache import cache_key, file_digest, load_cache, save_cache


def completed_job():
    return {**main.new_job(), 'status': 'completed', 'progress': 100,
            'results': [{'timestamp': 0, 'frame_index': 0, 'predictions': [],
                         'classification': {'status': 'normal', 'accidents': [], 'primary': None}}]}


def test_cached_job_survives_memory_reset_and_skips_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'CACHE_DIR', tmp_path / 'cache')
    source = tmp_path / 'video.mp4'
    source.write_bytes(b'original-video')
    digest = file_digest(source)
    save_cache(main.CACHE_DIR, digest, main.cache_settings(), completed_job())
    video_id = 'cache-restart-test'
    main.videos[video_id] = {'path': source, 'sha256': digest}
    main.jobs.pop(video_id, None)
    tasks = BackgroundTasks()
    try:
        response = main.analyze_video(video_id, tasks)
        assert response['status'] == 'completed'
        assert response['source'] == 'cache'
        assert tasks.tasks == []
        assert main.jobs[video_id]['results'][0]['timestamp'] == 0
        main.jobs.pop(video_id)
        main.analyze_video(video_id, tasks, force=True)
        assert main.jobs[video_id]['status'] == 'analyzing'
        assert len(tasks.tasks) == 1
    finally:
        main.videos.pop(video_id, None)
        main.jobs.pop(video_id, None)


def test_cache_is_invalidated_for_different_content_or_settings(tmp_path):
    source = tmp_path / 'video.mp4'
    source.write_bytes(b'first')
    digest = file_digest(source)
    settings = main.cache_settings()
    save_cache(tmp_path, digest, settings, completed_job())
    assert load_cache(tmp_path, digest, settings)['source'] == 'cache'
    source.write_bytes(b'second')
    assert load_cache(tmp_path, file_digest(source), settings) is None
    for key, value in [('threshold', .9), ('sample_fps', 1), ('model_id', 'different/1')]:
        assert load_cache(tmp_path, digest, {**settings, key: value}) is None


def test_corrupt_or_partial_cache_is_a_miss(tmp_path):
    settings = main.cache_settings()
    target = tmp_path / f"{cache_key('digest', settings)}.json"
    target.write_text('{incomplete', encoding='utf-8')
    assert load_cache(tmp_path, 'digest', settings) is None


def test_restored_cache_reclassifies_raw_predictions_with_current_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'CACHE_DIR', tmp_path)
    job = completed_job()
    job['results'][0]['predictions'] = [
        {'class': 'car_person_accident', 'confidence': .7},
        {'class': 'car_car_accident', 'confidence': .95},
        {'class': 'car', 'confidence': .99},
    ]
    save_cache(tmp_path, 'digest', main.cache_settings(), job)
    restored = main.cached_analysis('digest')
    result = restored['results'][0]['classification']
    assert result['primary']['class_id'] == 1
    assert result['primary']['confidence'] == .95
    assert {item['class_id'] for item in result['accidents']} == {1, 4}
    assert len(restored['events']) == 2
