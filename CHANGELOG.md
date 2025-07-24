# Changelog

## Refactor UI modules

- Split `cli/ui.py` into modular files: `audio_utils.py`, `file_utils.py`, `chat_agent.py`, `ui_components.py`, and `config.py`.
- Added lazy OpenAI initialization and improved error logging.
- Fixed audio recorder cancel behaviour and dropdown refresh after file operations.
- Added unit test `tests/test_audio_utils.py` verifying recorder cancel logic.
