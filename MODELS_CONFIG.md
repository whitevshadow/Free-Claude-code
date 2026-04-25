# Hierarchical Model Configuration

This file configures **automatic model fallback** for your Free Claude Code proxy.

## How It Works

When a request comes in for a Claude model (Opus/Sonnet/Haiku), the proxy:

1. **Identifies the tier** (Opus, Sonnet, or Haiku)
2. **Tries models in order** from the configured list
3. **Automatically falls back** if a model fails (rate limit, timeout, error)
4. **Logs the attempt** for debugging

## Configuration Structure

```json
{
  "model_tiers": {
    "opus": [
      {
        "name": "Human-readable name",
        "provider": "nvidia_nim",
        "model": "owner/model-name",
        "description": "What this model is good at"
      }
    ],
    "sonnet": [ ... ],
    "haiku": [ ... ]
  },
  "default_model": { ... }
}
```

## Current Configuration

### 🔴 Opus Tier (Most Capable)
1. **DeepSeek V4 Pro** (685B MoE) - Best reasoning and long context
2. **DeepSeek V4 Flash** (284B MoE) - Fast coding and agents
3. **Qwen 3.5 397B** (400B MoE) - Advanced vision and agentic capabilities
4. **Mistral Large 3** (675B MoE) - Chat and instruction following

### 🟡 Sonnet Tier (Balanced)
1. **Devstral-2** (123B) - Code model with deep reasoning
2. **Kimi K2.5** (1T MoE) - Multimodal understanding
3. **Kimi K2 Instruct** - Strong reasoning and coding
4. **Qwen3 Coder 480B** - Agentic coding with 256K context

### 🟢 Haiku Tier (Fast)
1. **GLM-4.7** - Fast general purpose
2. **Mistral Medium 3** - Enterprise multimodal
3. **Mistral Nemotron** - Agentic workflows

## Fallback Behavior

If a model fails (rate limit, API error, timeout), the system automatically tries the next model in the list:

```
Opus Request → DeepSeek V4 Pro ❌ Failed
             → DeepSeek V4 Flash ❌ Failed  
             → Qwen 3.5 397B ✅ Success
```

If all Opus models fail, it falls back to Sonnet tier, then Haiku tier.

## Customization

### Add a New Model

```json
{
  "name": "My Custom Model",
  "provider": "nvidia_nim",
  "model": "owner/model-name",
  "description": "What it's good at"
}
```

### Change Priority

Just reorder the array - the first model is tried first.

### Use Different Providers

```json
{
  "name": "OpenRouter Model",
  "provider": "open_router",
  "model": "anthropic/claude-3-opus",
  "description": "Official Claude via OpenRouter"
}
```

Valid providers: `nvidia_nim`, `open_router`, `deepseek`, `lmstudio`, `llamacpp`

## Finding NVIDIA NIM Model IDs

Visit https://build.nvidia.com/explore/discover and browse models.

Model ID format: `{owner}/{model-name}`

Examples:
- `deepseek-ai/deepseek-v4-pro`
- `mistralai/mistral-large-3-675b-instruct-2512`
- `qwen/qwen3.5-397b-a17b`

## Testing

After updating `models_config.json`, restart Docker:

```bash
docker-compose restart
```

Test with Claude Code CLI:
```powershell
$env:ANTHROPIC_BASE_URL = "http://localhost:8082"
$env:ANTHROPIC_AUTH_TOKEN = "root"
claude
```

Check logs for model selection:
```bash
docker-compose logs -f | grep "MODEL RESOLUTION"
```

## Backwards Compatibility

The `.env` file `MODEL_OPUS`, `MODEL_SONNET`, `MODEL_HAIKU` still work as fallbacks if `models_config.json` is missing or incomplete.

Priority: **JSON > ENV VARS > Default**

## Performance Tips

1. **Put fastest models first** if speed matters more than quality
2. **Put cheapest models first** if cost matters more than performance  
3. **Mix providers** for better availability (NVIDIA NIM, OpenRouter, DeepSeek)
4. **Test fallback chains** by intentionally causing failures

## Monitoring

Watch for fallback events:
```bash
docker-compose logs -f | grep "FALLBACK"
```

Example output:
```
FALLBACK: opus tier attempt 1: switching to DeepSeek V4 Flash (nvidia_nim/deepseek-ai/deepseek-v4-flash)
```

## Troubleshooting

**Models not failing over:**
- Check `FALLBACK_ROUTING=true` in `.env`
- Verify model IDs are correct
- Check NVIDIA NIM API key is valid

**All models failing:**
- Verify `NVIDIA_NIM_API_KEY` in `.env`
- Check rate limits: https://build.nvidia.com/settings/api-keys
- Test individual models at https://build.nvidia.com/explore/discover

**JSON not loading:**
- Check file is named exactly `models_config.json`
- Check file is in project root (same dir as `docker-compose.yml`)
- Validate JSON syntax: https://jsonlint.com/

## Complete Model List (46 models)

See `nvidia_nim_models.json` for the full catalog of available models.

Popular choices by use case:

**Coding:**
- `deepseek-ai/deepseek-v4-pro`
- `mistralai/devstral-2-123b-instruct-2512`
- `qwen/qwen3-coder-480b-a35b-instruct`

**Reasoning:**
- `deepseek-ai/deepseek-v3.2`
- `moonshotai/kimi-k2-thinking`
- `openai/gpt-oss-120b`

**Speed:**
- `deepseek-ai/deepseek-v4-flash`
- `z-ai/glm4.7`
- `mistralai/mistral-nemotron`

**Multimodal:**
- `qwen/qwen3.5-397b-a17b`
- `moonshotai/kimi-k2.5`
- `mistralai/mistral-large-3-675b-instruct-2512`

---

**Questions?** Check the logs: `docker-compose logs -f`
