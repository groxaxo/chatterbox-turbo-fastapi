"""ASGI entrypoint that layers English/Spanish profile routing over server.py."""

import server
from multilingual_runtime import install_multilingual_runtime

runtime = install_multilingual_runtime(server)
app = server.app
