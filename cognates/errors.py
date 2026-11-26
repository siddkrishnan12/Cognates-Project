"""The single error type the pipeline raises when a step must stop and report."""


class PipelineStop(Exception):
    """A step cannot continue. The message names the step and the reason."""

    def __init__(self, step: int, reason: str):
        self.step = step
        self.reason = reason
        super().__init__(f"Step {step}: {reason}")
