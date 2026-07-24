import time

import pytest
from fastapi import HTTPException

import backend


@pytest.fixture(autouse=True)
def clear_jobs():
    with backend._jobs_lock:
        backend._jobs.clear()
    yield
    with backend._jobs_lock:
        backend._jobs.clear()


def completed_job(job_id: str, finished_at: float) -> backend.AnalysisJob:
    return backend.AnalysisJob(
        job_id=job_id,
        prompt="prompt",
        expected_answer="answer",
        state="completed",
        finished_at=finished_at,
    )


def test_cleanup_removes_expired_jobs_and_limits_retained_jobs():
    now = time.monotonic()
    expired = completed_job("expired", now - backend.JOB_TTL_SECONDS - 1)
    retained = [
        completed_job(f"recent-{index}", now - index)
        for index in range(backend.MAX_RETAINED_JOBS + 2)
    ]

    with backend._jobs_lock:
        backend._jobs[expired.job_id] = expired
        backend._jobs.update({job.job_id: job for job in retained})
        backend.cleanup_jobs_locked()

        assert "expired" not in backend._jobs
        assert len(backend._jobs) == backend.MAX_RETAINED_JOBS
        assert "recent-0" in backend._jobs
        assert f"recent-{len(retained) - 1}" not in backend._jobs


def test_input_node_has_no_detail():
    job = backend.AnalysisJob(
        job_id="input-only",
        prompt="prompt",
        expected_answer="answer",
        nodes={
            "Input": backend.NodeRecord(
                kind="input",
                ready=True,
                rank=50257,
            )
        },
    )
    with backend._jobs_lock:
        backend._jobs[job.job_id] = job

    with pytest.raises(HTTPException) as exc_info:
        backend.node_detail(job.job_id, "Input")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "node has no detail"


def test_analyze_rejects_when_queue_is_full():
    acquired_slots = 0
    try:
        for _ in range(backend.MAX_PENDING_JOBS):
            assert backend._job_slots.acquire(blocking=False)
            acquired_slots += 1

        with pytest.raises(HTTPException) as exc_info:
            backend.analyze(
                backend.AnalyzeRequest(
                    prompt="prompt",
                    expected_answer="answer",
                )
            )

        assert exc_info.value.status_code == 429
    finally:
        for _ in range(acquired_slots):
            backend._job_slots.release()


def test_demo_input_normalization_only_strips_surrounding_whitespace():
    assert backend.normalize_demo_input("  The capital  of France is  ") == (
        "The capital  of France is"
    )


def test_analyze_stores_the_normalized_values_shown_by_the_frontend(monkeypatch):
    submitted_job_ids = []

    def record_submission(_function, job_id):
        submitted_job_ids.append(job_id)

    monkeypatch.setattr(backend._analysis_executor, "submit", record_submission)

    try:
        response = backend.analyze(
            backend.AnalyzeRequest(
                prompt="  The capital of France is  ",
                expected_answer="  Paris  ",
            )
        )
        job = backend.snapshot_job(response.job_id)

        assert submitted_job_ids == [response.job_id]
        assert job.prompt == "The capital of France is"
        assert job.expected_answer == "Paris"
    finally:
        backend._job_slots.release()
