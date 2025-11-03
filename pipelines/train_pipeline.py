"""
pipelines/train_pipeline.py

Lightweight pipeline runner that composes step callables or objects
(with a .run method) and executes them sequentially.

Behavior:
- If a step callable (or step.run) accepts >= 1 positional argument,
  the pipeline calls it with the previous step's output.
- Otherwise it calls it with no arguments.

This file has no external dependencies (ZenML-free).
"""
import inspect
import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainPipeline:
    def __init__(self, steps: List[Any], name: Optional[str] = None):
        """
        steps: list of callables or objects with a callable `run` attribute.
        name: optional pipeline name used in logs.
        """
        self.steps = list(steps)
        self.name = name or "train_pipeline"
        self.outputs = {}

    def _resolve_callable(self, step: Any):
        """
        Return a callable to execute for the step and a human-friendly name.
        """
        if hasattr(step, "run") and callable(getattr(step, "run")):
            func = getattr(step, "run")
            step_name = f"{step.__class__.__name__}.run"
        elif callable(step):
            func = step
            step_name = getattr(step, "__name__", repr(step))
        else:
            raise TypeError(f"Step {step} is neither callable nor has a callable 'run' method.")
        return func, step_name

    def _expects_input(self, func) -> bool:
        """
        Determine whether to pass previous output to func by inspecting
        its signature for positional params.
        """
        try:
            sig = inspect.signature(func)
            params = [
                p for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            return len(params) >= 1
        except Exception:
            # If signature inspection fails, assume we should try passing prev output
            return True

    def _call_step(self, step: Any, prev_output: Any):
        func, step_name = self._resolve_callable(step)
        expects = self._expects_input(func)
        logger.info(f"Executing step `{step_name}`; expects_input={expects}")
        if expects:
            return func(prev_output)
        return func()

    def run(self) -> dict:
        """
        Execute all steps sequentially and return a mapping of step index -> {name, output}.
        """
        logger.info(f"Starting pipeline '{self.name}' with {len(self.steps)} steps.")
        prev = None
        for idx, step in enumerate(self.steps):
            try:
                out = self._call_step(step, prev)
                # Save step name and output reference (not the full object to avoid huge JSON)
                _, step_name = self._resolve_callable(step)
                self.outputs[idx] = {"name": step_name, "output": out}
                prev = out
                logger.info(f"Step[{idx}] '{step_name}' completed. Output type: {type(out)}")
            except Exception as e:
                logger.exception(f"Step[{idx}] failed: {e}")
                raise
        logger.info(f"Pipeline '{self.name}' finished successfully.")
        return self.outputs


def train_pipeline(*steps: Any, name: Optional[str] = None) -> TrainPipeline:
    """
    Factory helper for constructing a TrainPipeline.
    Example:
        pipeline = train_pipeline(step1, step2, step3)
        pipeline.run()
    """
    return TrainPipeline(list(steps), name=name)
