"""ASGI entrypoint for official English Turbo plus chaturbo-espanol profiles."""

import server
from chaturbo_espanol_runtime import install_chaturbo_espanol_runtime
from performance_runtime import install_turbo_performance_runtime

runtime = install_chaturbo_espanol_runtime(server)
performance_runtime = install_turbo_performance_runtime(server, runtime)
app = server.app
