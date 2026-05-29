# Metrics Collection

## Purpose

Per-request server-side metrics reporting including timing, GPU assignment, and cache status.

## Requirements

### Requirement: Per-request server-side metrics

Every successful response from the `/edit` endpoint SHALL include a `metrics` object containing: `request_id` (UUID string), `gpu_id` (integer), `queue_latency_s` (float, seconds spent waiting for a free GPU), `service_time_s` (float, seconds spent on inference including VAE encoding if needed), `vae_encode_time_s` (float, seconds spent on VAE encoding or 0.0 if cache hit), `total_time_s` (float, total server-side processing time), `num_steps` (integer, denoising steps used), `cache_hit` (boolean, whether VAE latent cache was used), and `image_hash` (string, SHA-256 hex of the source image).

#### Scenario: Metrics on uncontended request

- **WHEN** a request is served immediately with no queuing
- **THEN** the response `metrics.queue_latency_s` is near zero and `metrics.service_time_s` reflects the actual inference duration

#### Scenario: Metrics distinguish cache hit from miss

- **WHEN** a request reuses a cached VAE latent
- **THEN** the response `metrics.cache_hit` is `true` and `metrics.vae_encode_time_s` is 0.0

#### Scenario: Metrics on cold-start request

- **WHEN** a request requires VAE encoding (no cached latent)
- **THEN** the response `metrics.cache_hit` is `false` and `metrics.vae_encode_time_s` reflects the actual VAE encoding duration

#### Scenario: Metrics on queued request

- **WHEN** a request waits 2.5 seconds for a free GPU before inference begins
- **THEN** the response `metrics.queue_latency_s` is approximately 2.5 and `metrics.service_time_s` reflects only the inference duration (excluding the wait)

#### Scenario: GPU ID is reported

- **WHEN** a request is processed on GPU 2
- **THEN** the response `metrics.gpu_id` is 2

### Requirement: Unique request identification

Every request SHALL be assigned a unique UUID `request_id` by the server. The `request_id` SHALL appear in the JSON response body under `metrics.request_id`.

#### Scenario: Two concurrent requests have different IDs

- **WHEN** two requests are submitted simultaneously
- **THEN** each response contains a distinct `request_id` value

### Requirement: Timing precision

Queue latency and service time SHALL be measured using a monotonic clock (e.g., `time.monotonic()`). Values SHALL be rounded to 3 decimal places (millisecond precision).

#### Scenario: Monotonic timing

- **WHEN** the system clock is adjusted during a request
- **THEN** the reported `queue_latency_s` and `service_time_s` are unaffected because they use a monotonic clock source

### Requirement: Image hash in response

Every successful response SHALL include `metrics.image_hash` containing the SHA-256 hex string of the source image bytes. This allows clients to reference the same image in subsequent requests without re-uploading it.

#### Scenario: Hash returned on first request

- **WHEN** a client uploads an image without providing `image_hash`
- **THEN** the response `metrics.image_hash` contains the SHA-256 hex string computed from the uploaded image bytes, which the client can use in subsequent requests
