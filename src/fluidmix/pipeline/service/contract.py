"""Pinned frontend runner contract constants."""

PIPELINE_COMMIT = "cf6829b99486b41f4190874dbd701ad812648217"
API_VERSION = "v1"
SERVICE_NAME = "fluidmix-pipeline-runner"
MAX_UPLOAD_BYTES = 536_870_912
UPLOAD_MIME_TYPES = ("video/mp4", "video/x-msvideo")
EVENT_TYPES = frozenset({
    "measuring", "waiting_for_references", "vial_decision", "model_vote",
    "averaged_decision", "explanation", "error", "rate_health",
})
RUNNER_API_SHA256 = "68abaf410fbabb09a2e3da471e6bc6c7aec464a0dae59c64db2459a5f22128f7"
EVENT_SCHEMA_SHA256 = "3ea09963177d3960e2a49f61958da0a899a5e53cf92adfbe971e902b7b92ccdd"
