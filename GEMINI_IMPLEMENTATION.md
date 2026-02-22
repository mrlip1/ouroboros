# Gemini OAuth Implementation

## Overview
Implemented native Gemini API support with OAuth authentication for the ouroboros project. The system tries Gemini first for google/ models, then falls back to OpenRouter if unavailable.

## Authentication Priority
1. **Colab Authentication** (Primary for Colab environment)
   - Uses `google.colab.auth.authenticate_user()`
   - Non-interactive, works in Google Colab notebooks
   
2. **gcloud ADC** (Application Default Credentials)
   - Uses existing gcloud CLI authentication
   - No interactive browser login required
   
3. **API Key** (Fallback)
   - Environment variables: `GEMINI_API_KEY` or `GOOGLE_API_KEY`
   
4. **OpenRouter** (Final fallback)
   - Used when Gemini is unavailable or fails
   - Paid service, requires `OPENROUTER_API_KEY`

## Key Features
- **Automatic detection**: Detects Colab environment and google/ model prefixes
- **Non-interactive**: No browser popups in Colab
- **Graceful fallback**: Seamlessly falls back to OpenRouter on failure
- **Zero cost**: Gemini API calls are free (via OAuth)
- **Tool support**: OpenRouter used when tools are needed (Gemini doesn't support tools yet)

## Modified Files
- `ouroboros/llm.py` - Main implementation
- `requirements.txt` - Added google-generativeai, google-auth

## Usage in Colab
```python
# In Google Colab, authentication happens automatically
# Just set your model to a google/ model:
import os
os.environ["OUROBOROS_MODEL"] = "google/gemini-2.0-flash-exp"

# The system will:
# 1. Call google.colab.auth.authenticate_user()
# 2. Use your Google account credentials
# 3. Make free API calls to Gemini
# 4. Fall back to OpenRouter if needed
```

## Usage Outside Colab
```bash
# Option 1: Use gcloud CLI
gcloud auth application-default login

# Option 2: Set API key
export GEMINI_API_KEY="your-api-key"

# Then run ouroboros normally
```

## Implementation Details
- Helper functions: `_is_google_model()`, `_extract_gemini_model_name()`, `_is_colab_environment()`
- New methods: `_try_configure_gemini()`, `_call_gemini_api()`
- Modified: `__init__()`, `chat()` methods
- Message format conversion: OpenAI format → Gemini format
- Usage tracking: Extracts token counts from Gemini responses

## Limitations
- Gemini doesn't support tool calling yet, so OpenRouter is used when tools are needed
- Only works for google/ prefixed models (e.g., google/gemini-2.0-flash-exp)
- Requires google-generativeai and google-auth packages

## Testing
```python
# Test in Python
from ouroboros.llm import LLMClient

client = LLMClient()
messages = [{"role": "user", "content": "Hello!"}]
response, usage = client.chat(messages, model="google/gemini-2.0-flash-exp")
print(response["content"])
print(f"Tokens: {usage['total_tokens']}, Cost: ${usage['cost']}")
```

## Backup
Original file backed up to: `ouroboros/llm.py.backup_before_gemini`
