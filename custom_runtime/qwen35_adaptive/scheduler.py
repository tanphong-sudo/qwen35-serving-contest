import os

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput

from .policy import apply_completion_cohort


class CompletionCohortAsyncScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decode_window = int(os.getenv("QWEN35_DECODE_WINDOW", "20"))
        if self.decode_window < 1:
            raise ValueError("QWEN35_DECODE_WINDOW must be positive")

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        apply_completion_cohort(
            self.running,
            self.decode_window,
            self.current_step,
        )
        return super().schedule(throttle_prefills=throttle_prefills)
