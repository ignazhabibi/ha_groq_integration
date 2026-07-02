# Groq Cloud Conversation

Custom Home Assistant integration for using the Groq Cloud API as a conversation
agent, AI task provider, speech-to-text provider, and text-to-speech provider.

The integration follows Home Assistant's LLM API patterns and uses Groq's
OpenAI-compatible API endpoint.

## Features

- Conversation agent for Home Assistant Assist.
- AI task entity for text and structured data generation.
- Vision support for AI tasks with image attachments.
- Speech-to-text entity for Groq Whisper transcription.
- Text-to-speech entity for Groq Orpheus WAV speech generation.
- `groq_cloud_conversation.generate_text` action for direct text generation
  from scripts and automations.
- Home Assistant LLM tool support through the Assist API.
- Groq Cloud access through the OpenAI-compatible API endpoint.
- Streaming response handling for conversation output and tool calls.

## Requirements

- Home Assistant 2026.6.0 or newer.
- A Groq Cloud API key from <https://console.groq.com/keys>.
- Network access from Home Assistant to `https://api.groq.com`.

## Installation

### HACS

1. Open HACS in Home Assistant.
2. Go to **Integrations**.
3. Open the three-dot menu and choose **Custom repositories**.
4. Add this repository URL and select **Integration** as the category.
5. Install **Groq Cloud Conversation**.
6. Restart Home Assistant.

### Manual

Copy the integration directory into your Home Assistant configuration:

```text
custom_components/groq_cloud_conversation
```

After copying, restart Home Assistant.

## Configuration

1. In Home Assistant, open **Settings > Devices & services**.
2. Select **Add integration**.
3. Search for **Groq Cloud Conversation**.
4. Enter your Groq API key.

The initial setup creates:

- one conversation subentry for Home Assistant Assist,
- one AI task subentry for text and structured data generation,
- one speech-to-text subentry for Assist pipeline transcription,
- one text-to-speech subentry for generated speech audio.

All subentries can be reconfigured from the integration options.

## Default Model

The default model is:

```text
llama-3.3-70b-versatile
```

You can override the model in each conversation subentry. The advanced
conversation options load available Groq Chat Completions models from the Groq
Models API and show them in a dropdown. Known models are labeled as Production
or Preview based on Groq's model documentation.

AI task model options are limited to Groq models with `json_schema` Structured
Outputs support. If an older AI task subentry has no compatible model stored,
structured tasks fall back to `openai/gpt-oss-20b`.

AI tasks with image attachments use the configured vision model. The default
vision model is `meta-llama/llama-4-scout-17b-16e-instruct`.

The default speech-to-text model is:

```text
whisper-large-v3-turbo
```

The speech-to-text subentry also supports `whisper-large-v3` from its advanced
options.

The default text-to-speech model is:

```text
canopylabs/orpheus-v1-english
```

The text-to-speech subentry can also be configured for
`canopylabs/orpheus-arabic-saudi`.

## Conversation Agent

The conversation entity can be selected as an Assist conversation agent. When a
Home Assistant LLM API is enabled for the subentry, the agent can use exposed
Home Assistant tools to control devices through Assist.

The default conversation prompt comes from Home Assistant's
`llm.DEFAULT_INSTRUCTIONS_PROMPT`. You can customize it in the conversation
subentry options.

## AI Tasks

The AI task entity supports regular text generation and structured data
generation. Structured tasks use Home Assistant's schema information and return
parsed JSON data. If a task includes image attachments, the integration sends
the images to a Groq vision-capable Chat Completions model.

## Actions

The integration provides a `groq_cloud_conversation.generate_text` action for
scripts and automations that need a direct Groq Chat Completions response. The
action returns response data with the generated `text`, selected `model`,
`finish_reason`, and Groq token `usage` when provided by the API.

## Speech-to-Text

The speech-to-text entity can be selected as an Assist pipeline STT provider.
It sends Home Assistant audio streams to Groq's OpenAI-compatible transcription
endpoint and returns the transcribed text to Assist.

## Text-to-Speech

The text-to-speech entity can be selected anywhere Home Assistant accepts a TTS
engine. It sends text to Groq's OpenAI-compatible speech endpoint and returns
WAV audio. Groq Orpheus currently limits input text to 200 characters.

## Development

Create a Python 3.14 environment and install the development dependencies:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Run the local checks:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
```

## License

MIT License. See [LICENSE](LICENSE).
