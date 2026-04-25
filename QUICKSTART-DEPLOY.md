# 🚀 Quick Deployment Reference

One-page cheat sheet for deploying Free Claude Code to the cloud.

---

## 📋 Pre-Deploy Checklist

- [ ] Get API key: [NVIDIA NIM](https://build.nvidia.com/settings/api-keys) | [OpenRouter](https://openrouter.ai/keys) | [DeepSeek](https://platform.deepseek.com/api_keys)
- [ ] Generate secure `ANTHROPIC_AUTH_TOKEN`: `openssl rand -hex 32`
- [ ] Fork/clone repository to your GitHub account
- [ ] Review [DEPLOYMENT.md](DEPLOYMENT.md) for detailed platform guides

---

## ⚡ Quick Commands

### **Render** (Easiest)
```bash
# 1. Connect GitHub repo in Render dashboard
# 2. Set environment variables
# 3. Deploy! 
# URL: https://your-app.onrender.com
```

### **Railway**
```bash
# 1. Connect GitHub repo in Railway dashboard
# 2. Add environment variables
# 3. Auto-deploy triggered
# URL: https://your-app.up.railway.app
```

### **Fly.io**
```bash
fly launch
fly secrets set NVIDIA_NIM_API_KEY="nvapi-xxxxx"
fly secrets set ANTHROPIC_AUTH_TOKEN="$(openssl rand -hex 32)"
fly deploy
# URL: https://your-app.fly.dev
```

### **Google Cloud Run**
```bash
gcloud run deploy free-claude-code \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars NVIDIA_NIM_API_KEY=nvapi-xxxxx
# URL: Provided in output
```

### **Docker Compose** (Self-Hosted)
```bash
git clone https://github.com/yourname/free-claude-code.git
cd free-claude-code
cp .env.example .env
nano .env  # Edit with your values
docker-compose up -d
# URL: http://localhost:8082
```

---

## 🔐 Required Environment Variables

**Minimum configuration:**
```bash
# Provider API Key (choose ONE)
NVIDIA_NIM_API_KEY="nvapi-xxxxx"

# Model
MODEL="nvidia_nim/moonshotai/kimi-k2-instruct"

# Security (CRITICAL!)
ANTHROPIC_AUTH_TOKEN="your-secret-token-here"
```

**Recommended for production:**
```bash
# All from above, plus:
CORS_ORIGINS="https://your-app.com"
MODEL_OPUS="nvidia_nim/z-ai/glm-4.7"
MODEL_SONNET="nvidia_nim/moonshotai/kimi-k2-instruct"
MODEL_HAIKU="nvidia_nim/stepfun-ai/step-3.5-flash"
ENABLE_CAVEMAN=true
PROVIDER_RATE_LIMIT=40
```

---

## 🧪 Test Your Deployment

```bash
# Health check
curl https://your-app.onrender.com/health
# Expected: {"status":"healthy"}

# Test API (with auth)
curl -X POST https://your-app.onrender.com/v1/messages \
  -H "x-api-key: your-token" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## 🔧 Configure Claude Code CLI

### **Windows PowerShell**
```powershell
$env:ANTHROPIC_BASE_URL="https://your-app.onrender.com/v1"
$env:ANTHROPIC_AUTH_TOKEN="your-secret-token"
claude
```

### **Linux/Mac**
```bash
export ANTHROPIC_BASE_URL="https://your-app.onrender.com/v1"
export ANTHROPIC_AUTH_TOKEN="your-secret-token"
claude
```

### **Permanent Configuration**

Add to `~/.bashrc` or `~/.zshrc`:
```bash
export ANTHROPIC_BASE_URL="https://your-app.onrender.com/v1"
export ANTHROPIC_AUTH_TOKEN="your-secret-token"
alias claude-free="claude"
```

---

## 💰 Cost Comparison

| Platform | Free Tier | Paid | SSL | Difficulty |
|----------|-----------|------|-----|------------|
| Render | ✅ (sleeps) | $7/mo | ✅ | ⭐ Easy |
| Railway | $5 credit | ~$10/mo | ✅ | ⭐ Easy |
| Fly.io | ✅ Limited | ~$5-15/mo | ✅ | ⭐⭐ Medium |
| Cloud Run | ✅ Generous | Pay-per-use | ✅ | ⭐⭐ Medium |
| AWS ECS | ❌ | ~$15-30/mo | Via ALB | ⭐⭐⭐ Hard |
| Self-Hosted | - | $5-20/mo | Certbot | ⭐⭐⭐ Hard |

**Recommendation:** Start with **Render** or **Railway** for easiest setup.

---

## 🐛 Common Issues

### "Container exits with code 1"
```bash
# Missing API key - set one of:
NVIDIA_NIM_API_KEY=xxx
OPENROUTER_API_KEY=xxx
DEEPSEEK_API_KEY=xxx
```

### "Health check failing"
```bash
# Check port configuration
PORT=8082  # Should match Dockerfile EXPOSE
```

### "401 Unauthorized"
```bash
# Set authentication token
ANTHROPIC_AUTH_TOKEN="your-secret-token"
```

### "CORS error"
```bash
# Configure CORS
CORS_ORIGINS="https://your-frontend.com"
```

---

## 📚 Full Documentation

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for:
- Detailed platform guides
- SSL/TLS setup
- Monitoring and alerting
- Advanced troubleshooting
- Production best practices

---

## 🆘 Need Help?

- **Documentation**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Issues**: https://github.com/whitevshadow/Free-Claude-code/issues
- **Discussions**: https://github.com/whitevshadow/Free-Claude-code/discussions
