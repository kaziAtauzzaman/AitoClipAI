"""Post-pipeline composition for feedback generation and persistence."""

import logging
from pathlib import Path

from pipeline import PrerecordedPipelineResult

from explainable_feedback.config import ExplainableFeedbackConfig
from explainable_feedback.contracts import ExplainableFeedbackReport
from explainable_feedback.generator import ExplainableFeedbackGenerator
from explainable_feedback.persistence import (
    FeedbackReportWriter,
    JsonExplainableFeedbackWriter,
)


class ExplainableFeedbackService:
    """Generate and persist feedback after a prerecorded pipeline run."""

    def __init__(
        self,
        config: ExplainableFeedbackConfig,
        generator: ExplainableFeedbackGenerator | None = None,
        writer: FeedbackReportWriter | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._generator = generator or ExplainableFeedbackGenerator(logger=logger)
        self._writer = writer or JsonExplainableFeedbackWriter(config.report_path)
        self._logger = logger or logging.getLogger(__name__)

    def create(
        self, result: PrerecordedPipelineResult
    ) -> tuple[ExplainableFeedbackReport, Path]:
        """Generate and persist one aggregate report."""

        report = self._generator.generate(
            result, schema_version=self._config.schema_version
        )
        path = self._writer.write(report)
        self._logger.info("feedback_report_written path=%s", path)
        return report, path
