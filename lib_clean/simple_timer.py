import time
from typing import Dict, List, Tuple

import torch


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

def print_durations_summary(key, durations_list: List):
    durations = torch.tensor(durations_list, dtype=torch.float32)
    mean, median, std, min, max = (
        format_duration_ns(time_ns) for time_ns in 
        (durations.mean(), durations.median(), durations.std(), durations.min(), durations.max())
    )
    print(f"{key}: Avg = {mean} +- {std}, Median = {median}, Min = {min}, Max = {max}")

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
        for key, durations_list in Timer._comitted_times.items():
            print_durations_summary(key, durations_list)

        Timer._comitted_times.clear()

class CudaTimer:
    # All times are in ns
    _events: Dict[str, Tuple[torch.cuda.Event, torch.cuda.Event]] = dict()
    _is_running: Dict[str, bool] = dict()
    _comitted_times: Dict[str, List[float]] = dict() 

    @staticmethod
    def register(key: str):
        if key not in CudaTimer._events:
            CudaTimer._events[key] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            CudaTimer._is_running[key] = False 
        
        assert CudaTimer._is_running[key] is False, 'Event already running'
        CudaTimer._is_running[key] = True
        # Make sure the op queue in cuda is empty 
        torch.cuda.synchronize()
        # Start recording 
        CudaTimer._events[key][0].record()
        
    @staticmethod
    def commit(key: str):
        assert CudaTimer._is_running.get(key, False) is True, 'Event not running'
        # Place event.record in queue 
        CudaTimer._events[key][1].record()
        # Now the event recording is executed 
        torch.cuda.synchronize()

        if key not in CudaTimer._comitted_times:
            CudaTimer._comitted_times[key] = list()

        start_event, end_event = CudaTimer._events[key]
        # The time is in millis, we want nanos 
        time_elapsed = start_event.elapsed_time(end_event) * 1e6
        CudaTimer._comitted_times[key].append(time_elapsed)
        CudaTimer._is_running[key] = False

    @staticmethod
    def print():
        ''' Print the average, min and max of every commited time ''' 
        for key, running in CudaTimer._is_running.items():
            assert not running, f'{key} still running'

        for key, durations_list in CudaTimer._comitted_times.items():
            print_durations_summary(key, durations_list)

        CudaTimer._comitted_times.clear()