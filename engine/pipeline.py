"""Pipeline identifiers saved with each job, including interrupted jobs."""

LEGACY_PIPELINE = "triposr-auto-finish-v1"
PIPELINE = "triposr-auto-finish-v2"
SHAPE_PIPELINE = "triposg-local-v1"
GUIDED_PIPELINE = "triposg-codex-guided-v1"
HOST_PIPELINE = "triposg-host-guided-v1"
GUIDED_PIPELINES = (GUIDED_PIPELINE, HOST_PIPELINE)
SHAPE_PIPELINES = (SHAPE_PIPELINE, *GUIDED_PIPELINES)
