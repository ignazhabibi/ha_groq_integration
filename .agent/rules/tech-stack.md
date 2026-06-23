---
trigger: always_on
---

# Tech Stack & Patterns

## Runtime

- Target Python is 3.14.
- Home Assistant compatibility starts at 2026.6.0.
- The integration domain is `groq_cloud_conversation`.
- Groq API access goes through the OpenAI-compatible API with `openai==2.21.0`.

## Home Assistant Integration

- Use config entries and config subentries only.
- Keep user-visible text in `strings.json` and translations.
- Use Home Assistant's shared async HTTPX client via
  `homeassistant.helpers.httpx_client.get_async_client(hass)`.
- Map setup and runtime failures to Home Assistant-native exceptions.

## Typing

- Strict mypy is enabled for `custom_components` and `tests`.
- Avoid `Any` unless a Home Assistant or OpenAI boundary makes it unavoidable.
- Use built-in generics such as `list[str]` and `dict[str, Any]`.
- Do not annotate `self`.

## Validation

- Use `pytest`, Ruff, and mypy as the core quality gates.
- For packaging or metadata changes, also run:

```bash
.venv/bin/python -m pip install -e . --dry-run
```

## Data Shapes

- Use Home Assistant and OpenAI SDK types at integration boundaries.
- Use `voluptuous` schemas and `voluptuous_openapi.convert` for HA LLM tool and
  structured-output schema conversion.
- Do not make live Groq API calls from tests.
