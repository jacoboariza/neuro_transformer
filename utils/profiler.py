import time
from typing import Callable, Optional, Sequence

import torch


class DeviceTimer:
    def __init__(self, device: torch.device):
        self.device = device

    def start(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(self, start_marker) -> float:
        if self.device.type == "cuda":
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize(self.device)
            return float(start_marker.elapsed_time(end_event))
        return float((time.perf_counter() - start_marker) * 1000.0)


def profile_callable_ms(fn: Callable[[], object], device: torch.device) -> tuple[object, float]:
    timer = DeviceTimer(device)
    start = timer.start()
    result = fn()
    elapsed_ms = timer.stop(start)
    return result, elapsed_ms


def estimate_flops_torch_profiler(
    fn: Callable[[], object],
    device: torch.device,
    warmup_runs: int = 1,
) -> float:
    for _ in range(max(0, warmup_runs)):
        fn()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, with_flops=True) as prof:
        fn()

    total_flops = 0.0
    for event in prof.key_averages():
        flops = getattr(event, "flops", None)
        if flops is not None:
            total_flops += float(flops)
    return total_flops


def estimate_module_flops(
    model: torch.nn.Module,
    model_inputs: Sequence[torch.Tensor],
    device: torch.device,
    kwargs: Optional[dict] = None,
) -> float:
    kwargs = kwargs or {}

    def _forward_only():
        with torch.no_grad():
            return model(*model_inputs, **kwargs)

    return estimate_flops_torch_profiler(_forward_only, device=device)
