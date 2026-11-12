import sys
import threading
import unittest
import weakref
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules_forge import main_thread


class SimulatedOOM(Exception):
    pass


class StopWorker(BaseException):
    pass


class Payload:
    pass


class MainThreadTests(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.unload = Mock()
        self.enterContext(patch.dict(sys.modules, {
            "backend.memory_management": SimpleNamespace(
                is_oom=lambda error: isinstance(error, SimulatedOOM),
                logger=self.logger,
            ),
            "modules.sd_models": SimpleNamespace(unload_model_weights=self.unload),
        }))
        self.enterContext(patch.multiple(
            main_thread,
            last_id=0,
            waiting_queue=deque(),
            finished_tasks={},
            last_exception=None,
        ))

    def test_oom_cleanup_finishes_before_next_queued_job(self):
        cleanup_started = threading.Event()
        allow_cleanup = threading.Event()
        next_job_started = threading.Event()
        payload_refs = []
        observations = []
        worker_errors = []

        def fail():
            payload = Payload()
            payload_refs.append(weakref.ref(payload))
            raise SimulatedOOM("allocation failed")

        def cleanup():
            observations.append(payload_refs[0]() is None)
            observations.append(sys.exception() is None)
            cleanup_started.set()
            if not allow_cleanup.wait(5):
                raise TimeoutError("Test did not release cleanup")

        def next_job():
            next_job_started.set()
            return "next result"

        def stop():
            raise StopWorker()

        def run_worker():
            try:
                main_thread.loop()
            except StopWorker:
                pass
            except BaseException as error:
                worker_errors.append(error)

        self.unload.side_effect = cleanup
        first_id = main_thread.async_run(fail)
        second_id = main_thread.async_run(next_job)
        worker = threading.Thread(target=run_worker)
        worker.start()
        try:
            self.assertTrue(cleanup_started.wait(5), "OOM did not trigger cleanup")
            self.assertFalse(next_job_started.is_set())
            with main_thread.condition:
                self.assertNotIn(first_id, main_thread.finished_tasks)
            self.assertEqual(observations, [True, True])
        finally:
            allow_cleanup.set()
            main_thread.async_run(stop)
            worker.join(5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(worker_errors, [])
        self.assertTrue(next_job_started.is_set())
        self.assertEqual(main_thread.finished_tasks[second_id].result, "next result")
        self.unload.assert_called_once_with()

    def test_cleanup_failure_is_reported_without_escaping_worker(self):
        self.unload.side_effect = RuntimeError("unload failed")
        task = main_thread.Task(1, Mock(side_effect=SimulatedOOM("allocation failed")), (), {})

        with patch.object(main_thread.traceback, "print_exc") as print_exc:
            task.work()

        self.assertIsNone(task.result)
        self.assertIn("OOM", main_thread.last_exception)
        self.assertIn("unload failed", main_thread.last_exception)
        self.unload.assert_called_once_with()
        self.assertTrue(self.logger.exception.called)
        print_exc.assert_called_once_with()

    def test_success_does_not_repeat_previous_oom_cleanup(self):
        main_thread.last_exception = "OOM"
        task = main_thread.Task(1, lambda: "result", (), {})

        task.work()

        self.assertEqual(task.result, "result")
        self.assertIsNone(main_thread.last_exception)
        self.unload.assert_not_called()

    def test_non_oom_error_does_not_repeat_previous_oom_cleanup(self):
        main_thread.last_exception = "OOM"
        task = main_thread.Task(1, Mock(side_effect=ValueError("invalid input")), (), {})
        with patch.object(main_thread.traceback, "print_exc") as print_exc:
            task.work()

        self.assertIsNone(task.result)
        self.assertEqual(main_thread.last_exception, "ValueError: invalid input")
        self.unload.assert_not_called()
        print_exc.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
