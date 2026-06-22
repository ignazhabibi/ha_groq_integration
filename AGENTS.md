# AGENTS.md

## Project Scope

This repository contains the `groq_cloud_conversation` custom integration for Home Assistant. Its purpose is to expose Groq Cloud through the OpenAI-compatible API as:

- a Home Assistant Assist conversation agent,
- an AI task entity for text and structured data generation,
- an adapter between Home Assistant's LLM API/tool system and Groq's OpenAI-compatible Chat Completions API.

Use Home Assistant's official LLM API docs as the behavioral contract. The official `openai_conversation` integration remains a useful lifecycle and Home Assistant integration reference, but Groq request/response transport should follow Groq's Chat Completions and tool-calling documentation:

- https://github.com/home-assistant/core/tree/dev/homeassistant/components/openai_conversation
- https://developers.home-assistant.io/docs/core/llm/
- https://console.groq.com/docs/tool-use/local-tool-calling
- https://console.groq.com/docs/structured-outputs

## Repository Layout

- `custom_components/groq_cloud_conversation/`: Home Assistant integration code.
- `tests/`: pytest tests using `pytest-homeassistant-custom-component`.
- `pyproject.toml`: dependency, Ruff, pytest, and strict mypy configuration.
- `hacs.json`: HACS metadata.
- `README.md`: user-facing project overview.

## Local Commands

Prefer Python 3.14, matching `pyproject.toml`.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format .
.venv/bin/python -m mypy
```

When changing only documentation, running the test suite is optional. For code changes, run the narrowest relevant tests first, then broaden to `pytest`, `ruff check`, and `mypy` when behavior, typing, or shared adapters are affected.

## Home Assistant Integration Rules

- Follow Home Assistant's config-entry-only pattern. Do not add YAML configuration paths.
- Keep the integration domain as `groq_cloud_conversation`.
- Keep Home Assistant objects and async lifecycle methods idiomatic: `async_setup_entry`, `async_unload_entry`, update listeners, platform forwarding, and config subentries.
- Use Home Assistant's shared async HTTP client via `homeassistant.helpers.httpx_client.get_async_client(hass)` for API clients.
- Map external API failures into Home Assistant-native errors:
  - setup authentication failures -> `ConfigEntryAuthFailed`,
  - setup transient API failures -> `ConfigEntryNotReady`,
  - runtime user-facing failures -> `HomeAssistantError`,
  - config-flow form errors -> `cannot_connect`, `invalid_auth`, or `unknown`.
- Keep translations and UI copy in `strings.json` when user-visible.
- Preserve subentry support for both `conversation` and `ai_task_data` unless a task explicitly changes that surface.

## LLM and Groq Rules

- Treat Home Assistant's LLM API docs as the contract. Conversation entities should call `chat_log.async_provide_llm_data(...)` with:
  - `user_input.as_llm_context(DOMAIN)`,
  - selected `CONF_LLM_HASS_API` values,
  - configured `CONF_PROMPT`,
  - `user_input.extra_system_prompt`.
- Pass Home Assistant LLM tools to Groq in the OpenAI-compatible Chat Completions `tools` shape.
- Keep conversation and tool-call handling streaming-aware. Preserve `chat_log.async_add_delta_content_stream(...)` and convert assistant/tool result content back into Chat Completions messages for follow-up iterations.
- Use non-streaming Chat Completions with `response_format: json_schema` for structured AI task output.
- Do not broaden control behavior in code or prompts. If an Assist bug involves wrong entities or room targets, inspect trace/config/history/exposure evidence before assuming the model or STT is at fault.
- Preserve structured-output behavior for AI tasks. If `task.structure` is set, return parsed JSON data or raise a clear `HomeAssistantError`.
- Keep model defaults centralized in `const.py`.

## Python Style

These conventions are adapted from the linked project rules and apply to all Python files in this repository.

- Every Python file starts with a purpose docstring.
- Public classes and methods need concise Google-style docstrings.
- Docstrings should explain the "what" and "why"; do not repeat types already present in signatures.
- Use f-strings for normal interpolation.
- Never use f-strings in logger calls. Use lazy logger interpolation, for example `_LOGGER.debug("Response: %s", value)`.
- Logger messages should not end with a period.
- Keep imports sorted.
- Prefer descriptive variable names over single-letter names.
- Use `snake_case` for variables and functions, `UPPER_CASE` for constants, and `_leading_underscore` for internal helpers.
- Boolean variables should start with `is_`, `has_`, or `should_` when practical.
- Use built-in generics such as `list[str]` and `dict[str, Any]`.
- Avoid `Any` where a precise type is reasonable; this codebase runs with strict mypy.
- Do not annotate `self`.
- Catch specific exceptions. Do not catch bare `Exception` unless matching an upstream Home Assistant pattern and there is no safer alternative.
- Comments should be rare, useful, full sentences, capitalized, and end with a period.

## Testing Rules

- Use `pytest` only; do not introduce `unittest.TestCase`.
- Keep test files named `test_*.py` and test functions named `test_*`.
- Mirror source modules where useful:
  - `config_flow.py` -> `tests/test_config_flow.py`
  - `conversation.py` -> `tests/test_conversation.py`
  - `ai_task.py` -> `tests/test_ai_task.py`
  - shared adapter behavior -> `tests/test_entity.py`
- Use `MockConfigEntry` from `pytest_homeassistant_custom_component.common` for integration setup.
- Mock Groq/OpenAI clients at the integration boundary. Do not make live Groq API calls in tests.
- Use realistic Home Assistant objects (`conversation.ChatLog`, `llm.APIInstance`, `llm.ToolInput`, config subentries) instead of testing only dict shims.
- Async tests should work with the configured `asyncio_mode = "auto"`; add explicit markers only if they help readability or compatibility.
- Prefer Arrange/Act/Assert structure for non-trivial tests. Add section comments only when they make the test easier to scan.
- For changes to streaming, tool calls, structured outputs, reauth, or config subentries, add or update focused regression tests.

## Review Checklist

Before finishing a code change:

- Confirm the change still matches Home Assistant's current LLM API expectations.
- Compare conversation-agent lifecycle changes with the official `openai_conversation` integration, while keeping Groq API transport aligned with Groq Chat Completions.
- Verify no secrets, API keys, or user-specific Home Assistant entity IDs are committed.
- Run the relevant tests and linters, or state clearly why they were not run.
- For Assist behavior fixes, validate the final Home Assistant conversation result, not only that a Groq request was made.
