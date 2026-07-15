"""ASGI entrypoint for official English Turbo plus chaturbo-espanol profiles."""

import server
from acceleration_runtime import install_acceleration_runtime
from chaturbo_espanol_runtime import install_chaturbo_espanol_runtime
from performance_runtime import install_turbo_performance_runtime

runtime = install_chaturbo_espanol_runtime(server)
performance_runtime = install_turbo_performance_runtime(server, runtime)
acceleration_runtime = install_acceleration_runtime(server, runtime, performance_runtime)
app = server.app
