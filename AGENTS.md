# AGENTS.md

This is the only agent instruction file for this repository. Do not add separate `.agent/` rule or workflow files.

## Scope

This repository contains the `groq_cloud_conversation` Home Assistant custom integration. It exposes Groq Cloud as:

- an Assist conversation agent,
- an AI task entity for text, structured data, and vision attachments,
- speech-to-text and text-to-speech entities,
- the `groq_cloud_conversation.generate_text` action,
- an adapter between Home Assistant's LLM/STT/TTS APIs and Groq's OpenAI-compatible API.

Important paths:

- `custom_components/groq_cloud_conversation/`: integration code.
- `tests/`: pytest coverage with `pytest-homeassistant-custom-component`.
- `.github/workflows/`: CI, stable release, and prerelease workflows.
- `README.md`, `hacs.json`, `pyproject.toml`, and `custom_components/groq_cloud_conversation/manifest.json`: user-facing and packaging metadata.

## Commands

Use Python 3.14.

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
.venv/bin/python -m pip install -e . --dry-run
```

For code changes, run focused tests first, then broaden to `pytest`, `ruff check`, and `mypy` when behavior, typing, or shared adapters are affected. Documentation-only changes do not require the full test suite. Run the dry-run install after packaging or metadata changes.

## Workflow

- Do not commit directly to `main` unless the user explicitly asks for it.
- Prefer short-lived branches and PRs for normal work.
- Pull requests and `main` pushes run Ruff, mypy, and pytest.
- Stable tags `vX.Y.Z` publish a GitHub release; prerelease tags `vX.Y.Z-alpha.N`, `vX.Y.Z-beta.N`, and `vX.Y.Z-rc.N` publish prereleases.
- Do not create release branches, bump versions, tag, push release refs, or claim a release is live without explicit user confirmation.
- Keep release versions aligned in `manifest.json` and `pyproject.toml`.
- HACS uses the standard repository layout under `custom_components/groq_cloud_conversation/`; do not add `zip_release` or release-asset filename requirements unless explicitly changing the distribution model.

## Integration Rules

- Follow Home Assistant's config-entry-only pattern. Do not add YAML configuration paths.
- Keep the domain `groq_cloud_conversation`.
- Preserve the `conversation`, `ai_task_data`, `stt`, and `tts` subentry surfaces unless a task explicitly changes them.
- Use Home Assistant's shared async HTTP client via `homeassistant.helpers.httpx_client.get_async_client(hass)`.
- Keep Groq transport in `api.py` on the local async `httpx`-based `GroqApiClient`; do not reintroduce the OpenAI SDK unless that dependency is deliberately added back.
- Map setup auth failures to `ConfigEntryAuthFailed`, setup transient failures to `ConfigEntryNotReady`, runtime user-facing failures to `HomeAssistantError`, and config-flow errors to `cannot_connect`, `invalid_auth`, or `unknown`.
- Keep user-visible UI copy in `strings.json` and translations.
- Keep model defaults in `const.py` and model capability routing in `model_registry.py`.
- Keep diagnostics redacting API keys, prompts, and system prompts.

## Groq and HA Behavior

- Home Assistant's LLM API is the behavioral contract. Use the official `openai_conversation` integration as a lifecycle reference, but keep Groq transport aligned with Groq Chat Completions, audio endpoints, tool-calling, and structured outputs.
- Conversation entities must call `chat_log.async_provide_llm_data(...)` with `user_input.as_llm_context(DOMAIN)`, selected `CONF_LLM_HASS_API`, configured `CONF_PROMPT`, and `user_input.extra_system_prompt`.
- Pass Home Assistant LLM tools to Groq in the Chat Completions `tools` shape.
- Preserve streaming conversation handling with `chat_log.async_add_delta_content_stream(...)` and follow-up tool iterations.
- Use non-streaming Chat Completions with `response_format: json_schema` for structured AI tasks.
- Route AI task image attachments to a vision-capable model; structured output with image attachments is intentionally unsupported.
- STT uses `/audio/transcriptions`; TTS uses `/audio/speech`.
- Do not broaden smart-home control behavior in code or prompts. For Assist entity or room mistakes, inspect traces, config, history, and exposure evidence first.

## Code and Tests

- Every Python file starts with a purpose docstring; public classes and methods need concise Google-style docstrings.
- Use lazy logger interpolation. Do not use f-strings in logger calls.
- Prefer precise types and built-in generics. Avoid `Any` where a reasonable precise type exists. Do not annotate `self`.
- Catch specific exceptions.
- Use `pytest` only. Do not introduce `unittest.TestCase`.
- Mock Groq API clients at the integration boundary. Do not make live Groq API calls in tests.
- Use realistic Home Assistant objects where practical, especially for conversation, LLM tools, config subentries, STT metadata, services, and diagnostics.
- Add or update focused regression tests for changes to streaming, tool calls, structured outputs, vision attachments, reauth, services, diagnostics, STT, TTS, model routing, or config subentries.

## Final Checks

- Verify no secrets, API keys, or user-specific Home Assistant entity IDs are committed.
- Validate edited JSON files such as `manifest.json`, `strings.json`, and translations with `python -m json.tool`.
- Run relevant tests and linters, or state clearly why they were not run.
- For Assist behavior fixes, validate the final Home Assistant conversation result, not only that a Groq request was made.
