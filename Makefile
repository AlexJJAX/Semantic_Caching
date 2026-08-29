REDIS_CONFIG ?= $(shell brew --prefix)/etc/redis.conf
REDIS_DATA_DIR ?= $(CURDIR)/.redis-data
WORKBENCH_HOST ?= 127.0.0.1
WORKBENCH_PORT ?= 8123
WORKBENCH_MODEL_MODE ?= live

.PHONY: setup doctor verify test-fast test-integration test-live test-live-web test-live-openai workbench redis-start redis-stop

setup:
	uv sync --locked

doctor:
	uv run portfolio-doctor

test-fast:
	uv run python -m unittest -v \
		tests.test_config \
		tests.test_phase2 \
		tests.test_phase2_requirements \
		tests.test_message_history.MessageHistoryUnitTests \
		tests.test_semantic_cache \
		tests.test_workbench.WorkbenchUnitTests

test-integration:
	uv run python -m unittest -v \
		tests.test_message_history.MessageHistoryRedisIntegrationTests \
		tests.test_redis_integration \
		tests.test_semantic_cache_redis \
		tests.test_vector_search_redis \
		tests.test_workbench.WorkbenchRedisIntegrationTests

test-live:
	RUN_LIVE_INTEGRATIONS=1 uv run python -m unittest tests.test_live_integrations -v

test-live-web:
	RUN_LIVE_WEB_TESTS=1 uv run python -m unittest \
		tests.test_live_integrations.LiveSourceWebsiteTests -v

test-live-openai:
	RUN_LIVE_OPENAI_TESTS=1 uv run python -m unittest \
		tests.test_live_integrations.LiveOpenAIRedisTests -v

verify:
	uv run ruff check .
	uv run python -m unittest discover -s tests -v
	uv run python -m compileall -q src RAG agentic evaluation llm_message_history semantic_cache vector_search workbench

workbench:
	WORKBENCH_MODEL_MODE="$(WORKBENCH_MODEL_MODE)" uv run python workbench/server.py --host "$(WORKBENCH_HOST)" --port "$(WORKBENCH_PORT)"

redis-start:
	mkdir -p "$(REDIS_DATA_DIR)"
	redis-server "$(REDIS_CONFIG)" \
		--daemonize yes \
		--dir "$(REDIS_DATA_DIR)" \
		--pidfile "$(REDIS_DATA_DIR)/redis.pid" \
		--logfile "$(REDIS_DATA_DIR)/redis.log"

redis-stop:
	redis-cli SHUTDOWN
