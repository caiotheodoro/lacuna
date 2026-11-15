"""Emit N numbered spans at a fixed pace through a default BatchSpanProcessor.

Every setting of the SDK is its default. The span's number is an attribute, so
the file the collector writes says exactly which spans arrived.
"""

from __future__ import annotations

import argparse
import time

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=600)
p.add_argument("--interval", type=float, default=0.05)
p.add_argument("--endpoint", default="http://localhost:4318/v1/traces")
a = p.parse_args()

provider = TracerProvider(resource=Resource.create({"service.name": "spike-emitter"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=a.endpoint)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("spike")

for i in range(a.n):
    with tracer.start_as_current_span("execute_tool") as span:
        span.set_attribute("spike.seq", i)
    print(i, flush=True)
    time.sleep(a.interval)
