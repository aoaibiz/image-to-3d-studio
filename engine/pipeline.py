"""Pipeline identifiers saved with each job, including interrupted jobs."""

SF3D_PIPELINE = "sf3d-official-preview"
SF3D_AGREEMENT = {
    "recipient": "Stability AI / Hugging Face Space stabilityai/stable-fast-3d",
    "image": "locally selected and segmented photograph", "quota": "personal anonymous ZeroGPU allowance",
    "purpose": "preview evaluation", "schema": 1,
}
LEGACY_PIPELINE = "triposr-auto-finish-v1"
PIPELINE = "triposr-auto-finish-v2"
SHAPE_PIPELINE = "triposg-local-v1"
GUIDED_PIPELINE = "triposg-codex-guided-v1"
HOST_PIPELINE = "triposg-host-guided-v1"
OBSERVED_PIPELINE = "triposg-codex-observed-v1"
OBSERVATION_METHODS = ("codex-photo-observations-v1", "codex-photo-observations-v2")
OBSERVED_AGREEMENT = {
    "schema": 1, "provider": "OpenAI Codex CLI / ChatGPT vision and image_generation",
    "recipient": "OpenAI ChatGPT", "personal_subscription": True, "api_key_billing": False,
    "send": ["normalized source photo for text observation",
             "normalized source photo and two rendered geometry views for inferred paint"],
    "max_photo_observations_per_job": 1, "max_image_generations_per_job": 1,
    "max_submissions_per_job": 2, "billing": "personal ChatGPT plan and account credit settings",
}
GUIDED_PIPELINES = (GUIDED_PIPELINE, HOST_PIPELINE, OBSERVED_PIPELINE)
SHAPE_PIPELINES = (SHAPE_PIPELINE, *GUIDED_PIPELINES)
