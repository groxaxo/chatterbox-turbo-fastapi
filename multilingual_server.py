"""ASGI entrypoint for official English Turbo plus chaturbo-espanol profiles."""

import server
from chaturbo_espanol_runtime import install_chaturbo_espanol_runtime

runtime = install_chaturbo_espanol_runtime(server)
app = server.app
