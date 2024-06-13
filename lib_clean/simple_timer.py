import time
from typing import Dict, List


def format_duration_ns(nanoseconds):
    units = [
        ("µs", 1_000),
        ("ms", 1_000_000),
        ("s", 1_000_000_000),
        ("min", 60 * 1_000_000_000),
        ("h", 60 * 60 * 1_000_000_000),
        ("d", 24 * 60 * 60 * 1_000_000_000),
    ]

    value = nanoseconds
    suffix = "ns"
    for unit, factor in units:
        if nanoseconds < factor:
            break
        value = nanoseconds / factor
        suffix = unit

    return f"{value:.1f}{suffix}"


class Timer:
    # All times are in ns
    _start_times: Dict[str, float] = dict()
    _comitted_times: Dict[str, List[float]] = dict()

    @staticmethod
    def register(key: str):
        assert key not in Timer._start_times, "Only one key per start time"
        Timer._start_times[key] = time.perf_counter_ns()

    @staticmethod
    def commit(key: str):
        assert key in Timer._start_times
        duration = time.perf_counter_ns() - Timer._start_times[key]

        if key not in Timer._comitted_times:
            Timer._comitted_times[key] = list()

        Timer._comitted_times[key].append(duration)
        del Timer._start_times[key]

    @staticmethod
    def print():
        """Print the average, min and max of every commited time"""
        assert len(Timer._start_times) == 0
        for key, durations in Timer._comitted_times.items():
            avg_time = sum(durations) / len(durations)
            min_time = min(durations)
            max_time = max(durations)
            print(
                f"{key}: Avg = {format_duration_ns(avg_time)}, Min = {format_duration_ns(min_time)}, Max = {format_duration_ns(max_time)}"
            )

        Timer._comitted_times.clear()
