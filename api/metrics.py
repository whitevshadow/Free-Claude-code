from collections import defaultdict
import time
from threading import Lock


class ModelMetrics:
    """Track model performance and fallback patterns"""

    def __init__(self):
        self.requests = defaultdict(int)
        self.failures = defaultdict(int)
        self.fallbacks = defaultdict(int)
        self.response_times = defaultdict(list)
        self.error_types = defaultdict(int)
        self.lock = Lock()

    def record_request(
        self, model: str, success: bool, response_time: float, is_fallback: bool = False
    ):
        with self.lock:
            key = f"{model}"
            self.requests[key] += 1
            self.response_times[key].append(response_time)
            if not success:
                self.failures[key] += 1
            if is_fallback:
                self.fallbacks[key] += 1

    def record_error(self, model: str, error_type: str):
        with self.lock:
            self.error_types[f"{model}:{error_type}"] += 1

    def get_stats(self):
        with self.lock:
            return {
                "total_requests": sum(self.requests.values()),
                "total_failures": sum(self.failures.values()),
                "total_fallbacks": sum(self.fallbacks.values()),
                "failure_rate": sum(self.failures.values())
                / max(sum(self.requests.values()), 1),
                "fallback_rate": sum(self.fallbacks.values())
                / max(sum(self.requests.values()), 1),
                "avg_response_times": {
                    model: sum(times) / len(times) if times else 0
                    for model, times in self.response_times.items()
                },
                "top_errors": sorted(
                    self.error_types.items(), key=lambda x: x[1], reverse=True
                )[:10],
            }
