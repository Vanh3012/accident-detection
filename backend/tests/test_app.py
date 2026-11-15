from pathlib import Path

from fastapi import HTTPException
import pytest

from app import main


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'CACHE_DIR', tmp_path / 'cache')

def pred(name, confidence=0.9, x=320, y=180, width=100, height=60):
    return {"class": name, "confidence": confidence, "x": x, "y": y, "width": width, "height": height}


def test_class_mapping_and_strongest_accident():
    result = main.classify([pred("car"), pred("bike_person_accident", .82), pred("car_car_accident", .94)])
    assert result["status"] == "accident"
    assert len(result["accidents"]) == 2
    assert result["primary"]["type"] == "vehicle_vehicle"
    assert result["primary"]["title"] == "Va chạm giữa các ô tô"


@pytest.mark.parametrize(("label", "category"), [
    ("car_car_accident", "vehicle_vehicle"),
    ("car_bike_accident", "motorcycle"),
    ("bike_bike_accident", "motorcycle"),
    ("bike_object_accident", "motorcycle"),
    ("car_object_accident", "single_vehicle"),
    ("car_person_accident", "pedestrian"),
    ("bike_person_accident", "pedestrian"),
])
def test_every_accident_label_maps_to_the_expected_class(label, category):
    assert main.classify([pred(label)])["primary"]["type"] == category


@pytest.mark.parametrize(('label', 'class_id'), [
    ('car_car_accident', 1), ('bike_object_accident', 2),
    ('car_object_accident', 3), ('bike_person_accident', 4),
])
def test_numeric_class_ids_match_dashboard(label, class_id):
    assert main.classify([pred(label)])['primary']['class_id'] == class_id


@pytest.mark.parametrize('confidence', [float('nan'), float('inf'), -1, 2, None, 'invalid'])
def test_invalid_confidence_never_triggers_alarm(confidence):
    assert main.classify([pred('car_car_accident', confidence)])['primary'] is None


def test_object_classes_do_not_trigger_accident():
    result = main.classify([pred("car", .99), pred("bike", .99), pred("person", .99)])
    assert result == {"status": "normal", "accidents": [], "primary": None}


def test_confidence_threshold_and_normal_state():
    assert main.classify([pred("car_bike_accident", .49)]) ["status"] == "normal"
    assert main.classify([pred("car_bike_accident", .5)]) ["primary"]["type"] == "motorcycle"


def test_bbox_conversion_respects_letterboxing():
    box = main.box_to_canvas(pred("car", x=960, y=540, width=200, height=100), 1920, 1080, 800, 600)
    assert box == pytest.approx({"x": 358.3333333333333, "y": 279.16666666666663,
                                 "width": 83.33333333333333, "height": 41.666666666666664})


def test_timestamp_and_event_deduplication():
    events = []
    accident = {"type": "motorcycle", "title": "Tai nạn liên quan đến xe máy", "confidence": .71, "label": "car_bike_accident"}
    main.record_event(events, accident, 2.0)
    main.record_event(events, {**accident, "confidence": .92, "label": "bike_bike_accident"}, 4.75)
    assert len(events) == 1
    assert events[0]["timestamp"] == 2.0
    assert events[0]["last_seen"] == 4.75
    assert events[0]["confidence"] == .92
    assert events[0]["labels"] == ["car_bike_accident", "bike_bike_accident"]
    main.record_event(events, accident, 8.0)
    assert len(events) == 2


def test_valid_upload_and_invalid_format(tmp_path, monkeypatch):
    assert main.validate_video_filename("traffic.mp4") == ".mp4"
    assert main.validate_video_filename("footage.MOV") == ".mov"
    with pytest.raises(HTTPException) as bad:
        main.validate_video_filename("notes.txt")
    assert bad.value.status_code == 400


def test_upload_size_limit_cleans_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 3)
    path = tmp_path / "partial.mp4"
    with path.open("wb") as out:
        assert main.write_upload_chunk(out, b"123", 0) == 3
        with pytest.raises(HTTPException) as response:
            main.write_upload_chunk(out, b"45", 3)
    assert response.value.status_code == 413
    path.unlink()
    assert list(Path(tmp_path).iterdir()) == []


def test_inference_failure_is_reported_as_error_not_normal(tmp_path, monkeypatch):
    video_id = "unit-test-video"
    video_path = tmp_path / "bad.mp4"
    video_path.write_bytes(b"bad")
    main.videos[video_id] = {"path": video_path, "filename": "bad.mp4", "size": 3}
    main.jobs[video_id] = {"status": "analyzing", "progress": 0, "processed_frames": 0,
                           "results": [], "events": [], "error": None, "duration": 0}
    monkeypatch.setattr(main, "_client", lambda: (_ for _ in ()).throw(RuntimeError("Roboflow unavailable")))
    main._run_analysis(video_id)
    assert main.jobs[video_id]["status"] == "error"
    assert main.jobs[video_id]["error"] == "Roboflow unavailable"
    assert main.jobs[video_id]["results"] == []
    main.videos.pop(video_id)
    main.jobs.pop(video_id)


def test_mock_roboflow_responses_are_kept_with_synchronized_timestamps(tmp_path, monkeypatch):
    video_id = "mock-inference-video"
    path = tmp_path / "source.mp4"
    path.write_bytes(b"mock")
    main.videos[video_id] = {"path": path, "filename": "source.mp4", "size": 4}
    main.jobs[video_id] = {"status": "analyzing", "progress": 0, "processed_frames": 0,
                           "results": [], "events": [], "error": None, "duration": 0}

    class Capture:
        def __init__(self, *_):
            self.index = 0
        def isOpened(self):
            return True
        def get(self, prop):
            return 4.0 if prop == main.cv2.CAP_PROP_FPS else 3
        def read(self):
            if self.index == 3:
                return False, None
            self.index += 1
            return True, object()
        def release(self):
            pass

    class FakeRoboflow:
        def infer(self, frame, model_id):
            assert model_id == main.MODEL_ID
            return {"predictions": [pred("car", .93), pred("car_car_accident", .88)]}

    monkeypatch.setattr(main.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(main, "_client", lambda: FakeRoboflow())
    main._run_analysis(video_id)
    job = main.jobs[video_id]
    assert job["status"] == "completed"
    assert [frame["timestamp"] for frame in job["results"]] == [0, .25, .5]
    assert all(len(frame["predictions"]) == 2 for frame in job["results"])
    assert len(job["events"]) == 1
    assert job["events"][0]["timestamp"] == 0
    main.videos.pop(video_id)
    main.jobs.pop(video_id)
