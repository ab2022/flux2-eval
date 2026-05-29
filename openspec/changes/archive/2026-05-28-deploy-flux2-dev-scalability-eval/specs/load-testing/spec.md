## ADDED Requirements

### Requirement: Concurrency sweep benchmark

The benchmark harness SHALL accept a comma-separated list of concurrency levels (e.g., `1,2,4,8,16`) and a total number of requests per level. It SHALL run all requests at each level before proceeding to the next level.

#### Scenario: Sweep across 4 levels

- **WHEN** the harness is invoked with `--concurrency 1,2,4,8 --requests-per-level 20`
- **THEN** it runs 20 requests at concurrency 1, then 20 at concurrency 2, then 20 at concurrency 4, then 20 at concurrency 8, in sequence

### Requirement: Per-request metric capture with cache fields

The harness SHALL record for each request: `request_id`, `concurrency_level`, `submit_time`, `response_time`, `total_latency_s` (client-side end-to-end), `queue_latency_s` (from server response), `service_time_s` (from server response), `vae_encode_time_s` (from server response), `cache_hit` (from server response), `gpu_id` (from server response), `status_code`, and `error` (empty string if none).

#### Scenario: Successful request metrics

- **WHEN** a request completes successfully
- **THEN** the harness records both client-side timing (total_latency_s) and server-side metrics (queue_latency_s, service_time_s, gpu_id) parsed from the response body

#### Scenario: Failed request metrics

- **WHEN** a request times out or returns a non-200 status
- **THEN** the harness records the error message and status code, with server-side metrics set to 0

### Requirement: Per-level summary statistics

After completing all requests at a given concurrency level, the harness SHALL compute and display: total requests, successful count, failed count, throughput (requests per second based on wall-clock time), latency percentiles (p50, p95, p99), mean queue latency, mean service time, GPU utilization spread (request count per GPU), cache hit count, cache miss count, and mean service time broken down by cache-hit vs cache-miss requests.

#### Scenario: Summary output at concurrency 4

- **WHEN** 20 requests complete at concurrency level 4
- **THEN** the harness prints a summary showing throughput in req/s, latency p50/p95/p99 in seconds, mean queue latency, mean service time, GPU spread, and cache hit/miss counts

#### Scenario: Summary shows cache speedup

- **WHEN** a related-request sweep completes with mixed cold-start and warm-cache requests
- **THEN** the summary separately reports mean service time for cache-hit requests and cache-miss requests, making the VAE encoding savings visible

### Requirement: Structured JSON output

The harness SHALL write all results to a JSON file specified by `--output`. The JSON SHALL contain one entry per concurrency level, each with a `summary` object and a `requests` array of per-request metrics.

#### Scenario: Output file contents

- **WHEN** the harness completes a sweep with levels 1,2,4
- **THEN** the output JSON file contains keys `"1"`, `"2"`, `"4"`, each with `summary` and `requests` fields

### Requirement: Multipart image upload

The harness SHALL send each request as a multipart form POST, including the test image file, the prompt string, and the `num_steps` parameter. The image file path and prompt text SHALL be specified via command-line arguments `--image` and `--prompt`.

#### Scenario: Image and prompt sent correctly

- **WHEN** the harness is invoked with `--image test.png --prompt "Make the sky blue"`
- **THEN** each request sends `test.png` as the `image` field and `"Make the sky blue"` as the `prompt` field in a multipart form POST

### Requirement: Related-request benchmark mode

The harness SHALL support a `--prompts-file` argument specifying a text file with one prompt per line. When provided, the harness SHALL send the same source image with each prompt in sequence, simulating a series of related edits to the same image. After the first request, subsequent requests SHALL include the `image_hash` returned by the server to enable VAE latent cache reuse.

#### Scenario: Related-request sweep with prompts file

- **WHEN** the harness is invoked with `--image hat_photo.png --prompts-file edits.txt` where `edits.txt` contains 10 prompts
- **THEN** the harness sends 10 requests using the same source image, with the first request uploading the image and subsequent requests including only `image_hash` and the next prompt

#### Scenario: Cache hit rate in related-request mode

- **WHEN** a related-request sweep completes with 10 prompts at concurrency 1
- **THEN** the per-level summary reports a cache hit rate of approximately 90% (9 out of 10 requests) and the mean service time for cache-hit requests is lower than the cold-start request

### Requirement: Request timeout

Each individual request SHALL have a timeout of 300 seconds. If a request exceeds this timeout, it SHALL be recorded as a failure.

#### Scenario: Slow request times out

- **WHEN** a request takes longer than 600 seconds
- **THEN** the harness records it as a failed request with an appropriate timeout error message
