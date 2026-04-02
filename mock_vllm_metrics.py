#!/usr/bin/env python3
"""
Mock vLLM Prometheus metrics server.

Emits realistic vLLM + DCGM metrics at :8000/metrics so you can develop
Grafana dashboards without a GPU. Metrics drift over time to simulate
a live inference workload.

Usage:
    python mock_vllm_metrics.py          # default port 8000
    python mock_vllm_metrics.py --port 8000 --dcgm-port 9400
"""

import argparse
import math
import random
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# ---------------------------------------------------------------------------
# Simulated state — evolves each tick (1s)
# ---------------------------------------------------------------------------
class SimState:
    def __init__(self):
        self.t0 = time.time()
        self.lock = threading.Lock()

        # Counters (monotonically increasing)
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self.request_success_stop = 0
        self.request_success_length = 0
        self.request_success_abort = 0
        self.prefix_cache_queries = 0
        self.prefix_cache_hits = 0
        self.num_preemptions = 0

        # Histogram accumulators
        self.e2e_latency_samples = []
        self.ttft_samples = []
        self.tpot_samples = []
        self.prefill_time_samples = []
        self.decode_time_samples = []
        self.queue_time_samples = []
        self.prompt_token_samples = []
        self.gen_token_samples = []

        # KV cache histograms
        self.kv_block_lifetime_samples = []
        self.kv_block_idle_samples = []

    def tick(self):
        """Simulate one second of inference activity."""
        with self.lock:
            elapsed = time.time() - self.t0

            # Simulate varying load (sinusoidal with noise)
            load_factor = 0.5 + 0.4 * math.sin(elapsed / 60) + random.uniform(-0.1, 0.1)
            load_factor = max(0.1, min(1.0, load_factor))

            rps = load_factor * 8  # ~0.8 to 8 requests/sec
            num_requests = max(1, int(rps + random.uniform(-1, 1)))

            for _ in range(num_requests):
                prompt_len = random.randint(50, 2000)
                gen_len = random.randint(20, 500)
                self.prompt_tokens += prompt_len
                self.generation_tokens += gen_len

                # Finish reason distribution
                r = random.random()
                if r < 0.85:
                    self.request_success_stop += 1
                elif r < 0.97:
                    self.request_success_length += 1
                else:
                    self.request_success_abort += 1

                # Prefix cache — hit rate improves over time (warm-up)
                self.prefix_cache_queries += 1
                hit_rate = min(0.7, 0.05 + elapsed / 600)
                if random.random() < hit_rate:
                    self.prefix_cache_hits += 1

                # Latency samples (seconds)
                prefill_ms = prompt_len * 0.015 + random.uniform(5, 30)
                decode_ms = gen_len * 3.5 + random.uniform(10, 50)
                queue_ms = (1 - load_factor) * 10 + random.uniform(0, 50 * load_factor)
                e2e_ms = prefill_ms + decode_ms + queue_ms
                ttft_ms = prefill_ms + queue_ms + random.uniform(1, 5)
                tpot_ms = decode_ms / max(gen_len, 1) * 1000  # per-token in ms

                self.e2e_latency_samples.append(e2e_ms / 1000)
                self.ttft_samples.append(ttft_ms / 1000)
                self.tpot_samples.append(tpot_ms / 1000)
                self.prefill_time_samples.append(prefill_ms / 1000)
                self.decode_time_samples.append(decode_ms / 1000)
                self.queue_time_samples.append(queue_ms / 1000)
                self.prompt_token_samples.append(prompt_len)
                self.gen_token_samples.append(gen_len)

            # KV cache eviction events (sporadic)
            if random.random() < 0.3 * load_factor:
                self.kv_block_lifetime_samples.append(random.uniform(0.5, 120))
                self.kv_block_idle_samples.append(random.uniform(0.1, 60))

            # Occasional preemptions under high load
            if load_factor > 0.7 and random.random() < 0.1:
                self.num_preemptions += 1

    def snapshot(self):
        with self.lock:
            elapsed = time.time() - self.t0
            load_factor = 0.5 + 0.4 * math.sin(elapsed / 60) + random.uniform(-0.05, 0.05)
            load_factor = max(0.1, min(1.0, load_factor))

            return {
                "elapsed": elapsed,
                "load_factor": load_factor,
                "prompt_tokens": self.prompt_tokens,
                "generation_tokens": self.generation_tokens,
                "request_success_stop": self.request_success_stop,
                "request_success_length": self.request_success_length,
                "request_success_abort": self.request_success_abort,
                "prefix_cache_queries": self.prefix_cache_queries,
                "prefix_cache_hits": self.prefix_cache_hits,
                "num_preemptions": self.num_preemptions,
                "e2e_latency": list(self.e2e_latency_samples),
                "ttft": list(self.ttft_samples),
                "tpot": list(self.tpot_samples),
                "prefill_time": list(self.prefill_time_samples),
                "decode_time": list(self.decode_time_samples),
                "queue_time": list(self.queue_time_samples),
                "prompt_token_hist": list(self.prompt_token_samples),
                "gen_token_hist": list(self.gen_token_samples),
                "kv_block_lifetime": list(self.kv_block_lifetime_samples),
                "kv_block_idle": list(self.kv_block_idle_samples),
            }


state = SimState()

# ---------------------------------------------------------------------------
# Prometheus histogram helper
# ---------------------------------------------------------------------------
LATENCY_BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64,
                   1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]

TOKEN_BUCKETS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000]

KV_BUCKETS = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 30, 60, 120, 300, 600, 1800]


def format_histogram(name, help_text, samples, buckets, labels=""):
    """Render a Prometheus histogram from raw samples."""
    if not samples:
        return ""

    lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
    sorted_buckets = sorted(buckets)
    count = len(samples)
    total = sum(samples)

    for b in sorted_buckets:
        le_count = sum(1 for s in samples if s <= b)
        lines.append(f'{name}_bucket{{le="{b}",{labels}}} {le_count}')
    lines.append(f'{name}_bucket{{le="+Inf",{labels}}} {count}')
    lines.append(f"{name}_count{{{labels}}} {count}")
    lines.append(f"{name}_sum{{{labels}}} {total:.6f}")
    return "\n".join(lines)


def format_counter(name, help_text, value, labels=""):
    return (
        f"# HELP {name} {help_text}\n"
        f"# TYPE {name} counter\n"
        f"{name}_total{{{labels}}} {value}\n"
        f"{name}_created{{{labels}}} {state.t0}"
    )


def format_gauge(name, help_text, value, labels=""):
    return (
        f"# HELP {name} {help_text}\n"
        f"# TYPE {name} gauge\n"
        f"{name}{{{labels}}} {value}"
    )


# ---------------------------------------------------------------------------
# vLLM metrics renderer
# ---------------------------------------------------------------------------
def render_vllm_metrics():
    s = state.snapshot()
    lbl = f'model_name="{MODEL_NAME}",engine="0"'

    parts = []

    # -- Gauges --
    running = int(s["load_factor"] * 24 + random.uniform(-2, 2))
    waiting = int((1 - s["load_factor"]) * 5 + random.uniform(0, 8 * s["load_factor"]))
    kv_usage = min(0.95, 0.3 + 0.5 * s["load_factor"] + random.uniform(-0.05, 0.05))

    parts.append(format_gauge("vllm:num_requests_running",
                              "Number of requests in model execution batches.", max(0, running), lbl))
    parts.append(format_gauge("vllm:num_requests_waiting",
                              "Number of requests waiting to be processed.", max(0, waiting), lbl))
    parts.append(format_gauge("vllm:kv_cache_usage_perc",
                              "KV-cache usage. 1 means 100 percent usage.", round(kv_usage, 4), lbl))

    # -- Counters --
    parts.append(format_counter("vllm:prompt_tokens",
                                "Number of prefill tokens processed.", s["prompt_tokens"], lbl))
    parts.append(format_counter("vllm:generation_tokens",
                                "Number of generation tokens processed.", s["generation_tokens"], lbl))
    parts.append(format_counter("vllm:num_preemptions",
                                "Cumulative number of preemption from the engine.", s["num_preemptions"], lbl))
    parts.append(format_counter("vllm:prefix_cache_queries",
                                "Number of prefix cache queries.", s["prefix_cache_queries"], lbl))
    parts.append(format_counter("vllm:prefix_cache_hits",
                                "Number of prefix cache hits.", s["prefix_cache_hits"], lbl))

    # Request success by finish reason
    for reason, val in [("stop", s["request_success_stop"]),
                        ("length", s["request_success_length"]),
                        ("abort", s["request_success_abort"])]:
        reason_lbl = f'{lbl},finished_reason="{reason}"'
        parts.append(format_counter("vllm:request_success",
                                    "Count of successfully processed requests.", val, reason_lbl))

    # -- Latency Histograms --
    parts.append(format_histogram("vllm:e2e_request_latency_seconds",
                                  "End-to-end request latency in seconds.",
                                  s["e2e_latency"], LATENCY_BUCKETS, lbl))
    parts.append(format_histogram("vllm:time_to_first_token_seconds",
                                  "Time to first token latency in seconds.",
                                  s["ttft"], LATENCY_BUCKETS, lbl))
    parts.append(format_histogram("vllm:inter_token_latency_seconds",
                                  "Inter-token latency in seconds.",
                                  s["tpot"], LATENCY_BUCKETS, lbl))
    parts.append(format_histogram("vllm:request_prefill_time_seconds",
                                  "Time spent in PREFILL phase.",
                                  s["prefill_time"], LATENCY_BUCKETS, lbl))
    parts.append(format_histogram("vllm:request_decode_time_seconds",
                                  "Time spent in DECODE phase.",
                                  s["decode_time"], LATENCY_BUCKETS, lbl))
    parts.append(format_histogram("vllm:request_queue_time_seconds",
                                  "Time spent in WAITING queue.",
                                  s["queue_time"], LATENCY_BUCKETS, lbl))

    # -- Token Histograms --
    parts.append(format_histogram("vllm:request_prompt_tokens",
                                  "Number of prefill tokens per request.",
                                  s["prompt_token_hist"], TOKEN_BUCKETS, lbl))
    parts.append(format_histogram("vllm:request_generation_tokens",
                                  "Number of generation tokens per request.",
                                  s["gen_token_hist"], TOKEN_BUCKETS, lbl))

    # -- KV Cache Residency Histograms --
    parts.append(format_histogram("vllm:kv_block_lifetime_seconds",
                                  "KV cache block lifetime from allocation to eviction.",
                                  s["kv_block_lifetime"], KV_BUCKETS, lbl))
    parts.append(format_histogram("vllm:kv_block_idle_before_evict_seconds",
                                  "Idle time before KV cache block eviction.",
                                  s["kv_block_idle"], KV_BUCKETS, lbl))

    # -- Cache config info --
    parts.append(format_gauge("vllm:cache_config_info",
                              "Information of the KV-cache config.",
                              1,
                              f'{lbl},block_size="16",cache_dtype="auto",'
                              f'gpu_memory_utilization="0.9",num_gpu_blocks="4096",'
                              f'num_cpu_blocks="512"'))

    return "\n\n".join(p for p in parts if p) + "\n"


# ---------------------------------------------------------------------------
# DCGM metrics renderer
# ---------------------------------------------------------------------------
def render_dcgm_metrics():
    s = state.snapshot()
    gpu_util = min(99, int(s["load_factor"] * 85 + random.uniform(0, 15)))
    mem_used = int(14000 + s["load_factor"] * 2000 + random.uniform(-200, 200))  # MiB
    mem_free = 81920 - mem_used
    temp = int(45 + s["load_factor"] * 25 + random.uniform(-3, 3))
    power = int(120 + s["load_factor"] * 180 + random.uniform(-10, 10))
    sm_active = s["load_factor"] * 0.8 + random.uniform(0, 0.15)

    gpu_lbl = 'gpu="0",UUID="GPU-mock-0000",device="nvidia_a100",modelName="NVIDIA A100-SXM4-80GB",' \
              'Hostname="vllm-node-0",DCGM_FI_DRIVER_VERSION="535.129.03"'

    lines = []

    def dcgm_gauge(name, help_text, value):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name}{{{gpu_lbl}}} {value}")

    dcgm_gauge("DCGM_FI_DEV_GPU_UTIL", "GPU utilization (%).", gpu_util)
    dcgm_gauge("DCGM_FI_DEV_FB_USED", "Frame buffer memory used (MiB).", mem_used)
    dcgm_gauge("DCGM_FI_DEV_FB_FREE", "Frame buffer memory free (MiB).", mem_free)
    dcgm_gauge("DCGM_FI_DEV_GPU_TEMP", "GPU temperature (C).", temp)
    dcgm_gauge("DCGM_FI_DEV_POWER_USAGE", "Power draw (W).", power)
    dcgm_gauge("DCGM_FI_PROF_SM_ACTIVE", "SM activity ratio.", round(sm_active, 4))
    dcgm_gauge("DCGM_FI_DEV_MEM_COPY_UTIL", "Memory controller utilization (%).",
               int(s["load_factor"] * 40 + random.uniform(0, 10)))
    dcgm_gauge("DCGM_FI_PROF_DRAM_ACTIVE", "DRAM active ratio.",
               round(s["load_factor"] * 0.6 + random.uniform(0, 0.1), 4))

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTTP servers
# ---------------------------------------------------------------------------
class VLLMMetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = render_vllm_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path in ("/health", "/ping"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # silence logs


class DCGMMetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = render_dcgm_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def run_ticker():
    while True:
        state.tick()
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Mock vLLM + DCGM Prometheus metrics server")
    parser.add_argument("--port", type=int, default=8000, help="vLLM metrics port (default: 8000)")
    parser.add_argument("--dcgm-port", type=int, default=9400, help="DCGM metrics port (default: 9400)")
    args = parser.parse_args()

    # Start background ticker
    ticker = threading.Thread(target=run_ticker, daemon=True)
    ticker.start()

    # Start DCGM server in background
    dcgm_server = HTTPServer(("0.0.0.0", args.dcgm_port), DCGMMetricsHandler)
    dcgm_thread = threading.Thread(target=dcgm_server.serve_forever, daemon=True)
    dcgm_thread.start()

    # Start vLLM server (foreground)
    vllm_server = HTTPServer(("0.0.0.0", args.port), VLLMMetricsHandler)
    print(f"Mock vLLM metrics at http://localhost:{args.port}/metrics")
    print(f"Mock DCGM metrics at http://localhost:{args.dcgm_port}/metrics")
    print("Metrics drift over time to simulate real workload. Ctrl+C to stop.")
    vllm_server.serve_forever()


if __name__ == "__main__":
    main()
