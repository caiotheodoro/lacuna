# The rig's own processes: the kernel service, the MCP fronts and the originators.
# Everything else in the rig is a released image, unmodified.
FROM python@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
COPY rig/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt
COPY src /app/src
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
WORKDIR /app
