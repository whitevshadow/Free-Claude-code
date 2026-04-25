# 🚀 Cloud Deployment Guide

Complete guide for deploying Free Claude Code to production on major cloud platforms.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Platform-Specific Guides](#platform-specific-guides)
  - [Render (Recommended)](#1-render-recommended)
  - [Railway](#2-railway)
  - [Fly.io](#3-flyio)
  - [AWS ECS/Fargate](#4-aws-ecsfargate)
  - [Google Cloud Run](#5-google-cloud-run)
  - [Azure Container Apps](#6-azure-container-apps)
  - [DigitalOcean App Platform](#7-digitalocean-app-platform)
  - [Self-Hosted Docker](#8-self-hosted-docker)
- [Environment Variables Checklist](#environment-variables-checklist)
- [SSL/TLS Configuration](#ssltls-configuration)
- [Monitoring & Observability](#monitoring--observability)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before deploying, ensure you have:

1. ✅ **API Keys** from your chosen provider(s):
   - NVIDIA NIM: https://build.nvidia.com/settings/api-keys
   - OpenRouter: https://openrouter.ai/keys
   - DeepSeek: https://platform.deepseek.com/api_keys
   - LM Studio/llama.cpp: No API key needed (local only)

2. ✅ **GitHub Account** (for most platforms)

3. ✅ **Payment Method** (most platforms offer free tiers)

---

## Platform-Specific Guides

### 1. Render (Recommended)

**Why Render?** Built-in SSL, automatic deployments, generous free tier, zero-config Docker support.

#### **Steps:**

1. **Fork/Clone Repository** to your GitHub account

2. **Create Web Service** on [Render](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the `free-claude-code` repository

3. **Configure Service**
   ```yaml
   Name: free-claude-code
   Region: Select closest to your users
   Branch: main
   Runtime: Docker
   Plan: Starter ($7/month) or Free (sleeps after 15 min)
   ```

4. **Set Environment Variables** (in Render dashboard)
   ```bash
   # Required: Choose ONE provider
   NVIDIA_NIM_API_KEY=nvapi-xxxxx
   # OR
   OPENROUTER_API_KEY=sk-or-xxxxx
   # OR
   DEEPSEEK_API_KEY=sk-xxxxx

   # Model Configuration
   MODEL=nvidia_nim/moonshotai/kimi-k2-instruct
   MODEL_OPUS=nvidia_nim/z-ai/glm-4.7
   MODEL_SONNET=nvidia_nim/moonshotai/kimi-k2-instruct
   MODEL_HAIKU=nvidia_nim/stepfun-ai/step-3.5-flash

   # Security (IMPORTANT for production!)
   ANTHROPIC_AUTH_TOKEN=your-secret-token-here
   CORS_ORIGINS=https://your-app.com

   # Performance
   PROVIDER_RATE_LIMIT=40
   PROVIDER_MAX_CONCURRENCY=5

   # Optional Features
   ENABLE_CAVEMAN=true
   ENABLE_THINKING=true
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Render automatically builds and deploys your Docker container
   - Note your service URL: `https://your-app.onrender.com`

6. **Configure Claude Code CLI**
   ```bash
   # Windows PowerShell
   $env:ANTHROPIC_BASE_URL="https://your-app.onrender.com/v1"
   $env:ANTHROPIC_AUTH_TOKEN="your-secret-token-here"
   claude

   # Linux/Mac
   export ANTHROPIC_BASE_URL="https://your-app.onrender.com/v1"
   export ANTHROPIC_AUTH_TOKEN="your-secret-token-here"
   claude
   ```

#### **Outbound IP Addresses**

If you need to whitelist Render's IPs in your provider's firewall or API settings:

```
Render Outbound IPs (Shared across services in the same region):
74.220.52.0/24
74.220.60.0/24
```

**Note**: These IPs are shared by all Render services in your region and are not unique to your deployment. Most API providers (NVIDIA NIM, OpenRouter, DeepSeek) don't require IP whitelisting.

#### **Render.yaml (Infrastructure as Code)**

Create `render.yaml` in your repository root for automated deployments:

```yaml
services:
  - type: web
    name: free-claude-code
    runtime: docker
    region: oregon
    plan: starter
    branch: main
    dockerfilePath: ./Dockerfile
    envVars:
      - key: NVIDIA_NIM_API_KEY
        sync: false
      - key: MODEL
        value: nvidia_nim/moonshotai/kimi-k2-instruct
      - key: ANTHROPIC_AUTH_TOKEN
        generateValue: true
      - key: ENABLE_CAVEMAN
        value: true
      - key: PROVIDER_RATE_LIMIT
        value: 40
    healthCheckPath: /health
```

**Cost**: Free tier (with sleep) or $7/month (always on)

---

### 2. Railway

**Why Railway?** Simple interface, great DX, generous free trial ($5 credit).

#### **Steps:**

1. **Sign up** at [Railway.app](https://railway.app)

2. **New Project** → "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects Dockerfile

3. **Add Environment Variables**
   - Click "Variables" tab
   - Add all required variables (see checklist below)

4. **Configure Domain**
   - Railway provides: `your-app.up.railway.app`
   - Or connect custom domain

5. **Deploy**
   - Railway automatically builds and deploys
   - Health check: `https://your-app.up.railway.app/health`

**Cost**: Free $5 credit, then pay-as-you-go (~$5-20/month)

---

### 3. Fly.io

**Why Fly.io?** Edge deployment, multiple regions, excellent CLI.

#### **Steps:**

1. **Install Fly CLI**
   ```bash
   # Windows (PowerShell)
   iwr https://fly.io/install.ps1 -useb | iex

   # Mac/Linux
   curl -L https://fly.io/install.sh | sh
   ```

2. **Authenticate**
   ```bash
   fly auth signup
   # or
   fly auth login
   ```

3. **Initialize App**
   ```bash
   cd free-claude-code
   fly launch
   ```

   - App name: `free-claude-code-yourname`
   - Region: Choose closest to you
   - PostgreSQL: No
   - Redis: No

4. **Configure Secrets**
   ```bash
   fly secrets set \
     NVIDIA_NIM_API_KEY="nvapi-xxxxx" \
     ANTHROPIC_AUTH_TOKEN="your-secret-token" \
     MODEL="nvidia_nim/moonshotai/kimi-k2-instruct"
   ```

5. **Deploy**
   ```bash
   fly deploy
   ```

6. **Monitor**
   ```bash
   fly logs
   fly status
   ```

**fly.toml** configuration:
```toml
app = "free-claude-code-yourname"
primary_region = "lax"

[build]

[http_service]
  internal_port = 8082
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  memory = '512mb'
  cpu_kind = 'shared'
  cpus = 1

[env]
  HOST = "0.0.0.0"
  PORT = "8082"

[[services]]
  protocol = "tcp"
  internal_port = 8082

  [[services.ports]]
    port = 80
    handlers = ["http"]
    force_https = true

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

  [services.concurrency]
    type = "connections"
    hard_limit = 25
    soft_limit = 20

  [[services.tcp_checks]]
    interval = "15s"
    timeout = "2s"
    grace_period = "10s"

  [[services.http_checks]]
    interval = "30s"
    timeout = "5s"
    grace_period = "10s"
    method = "get"
    path = "/health"
```

**Cost**: Free tier (2 VMs with auto-sleep), then ~$5-15/month

---

### 4. AWS ECS/Fargate

**Why AWS?** Enterprise-grade, scalable, integrates with AWS ecosystem.

#### **Option A: AWS Copilot (Easy)**

1. **Install AWS Copilot CLI**
   ```bash
   # Windows
   Invoke-WebRequest -OutFile 'C:\Program Files\copilot.exe' https://github.com/aws/copilot-cli/releases/latest/download/copilot-windows.exe

   # Mac
   brew install aws/tap/copilot-cli
   ```

2. **Initialize Application**
   ```bash
   copilot app init free-claude-code
   ```

3. **Create Service**
   ```bash
   copilot svc init --name proxy \
     --svc-type "Load Balanced Web Service" \
     --dockerfile ./Dockerfile \
     --port 8082
   ```

4. **Set Secrets**
   ```bash
   copilot secret init --name NVIDIA_NIM_API_KEY
   copilot secret init --name ANTHROPIC_AUTH_TOKEN
   ```

5. **Deploy**
   ```bash
   copilot env init --name production --profile default
   copilot deploy
   ```

#### **Option B: Manual ECS Setup**

1. **Create ECR Repository**
   ```bash
   aws ecr create-repository --repository-name free-claude-code
   ```

2. **Build & Push Docker Image**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

   docker build -t free-claude-code .
   docker tag free-claude-code:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/free-claude-code:latest
   docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/free-claude-code:latest
   ```

3. **Create ECS Cluster**
   - Go to AWS ECS Console
   - Create Cluster → Networking only (Fargate)

4. **Create Task Definition**
   - Launch type: Fargate
   - Task memory: 1GB
   - Task CPU: 0.5 vCPU
   - Container image: Your ECR image URL
   - Port: 8082
   - Environment variables: Add from checklist

5. **Create Service**
   - Cluster: Your cluster
   - Launch type: Fargate
   - Task definition: Your task
   - Number of tasks: 1
   - Load balancer: Application Load Balancer
   - Target group: Create new, port 8082
   - Health check path: `/health`

**Cost**: ~$15-30/month (Fargate + ALB)

---

### 5. Google Cloud Run

**Why Cloud Run?** Serverless, auto-scaling, generous free tier.

#### **Steps:**

1. **Install gcloud CLI**
   ```bash
   # Download from https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Build & Deploy**
   ```bash
   gcloud run deploy free-claude-code \
     --source . \
     --region us-central1 \
     --platform managed \
     --allow-unauthenticated \
     --port 8082 \
     --memory 1Gi \
     --timeout 300s \
     --set-env-vars NVIDIA_NIM_API_KEY=nvapi-xxxxx \
     --set-env-vars ANTHROPIC_AUTH_TOKEN=your-token \
     --set-env-vars MODEL=nvidia_nim/moonshotai/kimi-k2-instruct
   ```

4. **Set Secrets (Better Practice)**
   ```bash
   # Create secrets
   echo -n "nvapi-xxxxx" | gcloud secrets create nvidia-nim-key --data-file=-
   
   # Deploy with secrets
   gcloud run deploy free-claude-code \
     --source . \
     --region us-central1 \
     --set-secrets NVIDIA_NIM_API_KEY=nvidia-nim-key:latest
   ```

5. **Custom Domain**
   ```bash
   gcloud run domain-mappings create --service free-claude-code --domain api.yourdomain.com
   ```

**Cost**: Free tier (2 million requests/month), then pay-per-use

---

### 6. Azure Container Apps

**Why Azure?** Integrates with Microsoft ecosystem, good for enterprise.

#### **Steps:**

1. **Install Azure CLI**
   ```bash
   # Download from https://aka.ms/installazurecliwindows
   ```

2. **Login & Setup**
   ```bash
   az login
   az group create --name free-claude-code-rg --location eastus
   ```

3. **Create Container Registry**
   ```bash
   az acr create --resource-group free-claude-code-rg \
     --name freeclaudecode --sku Basic
   
   az acr login --name freeclaudecode
   ```

4. **Build & Push**
   ```bash
   docker build -t freeclaudecode.azurecr.io/free-claude-code:latest .
   docker push freeclaudecode.azurecr.io/free-claude-code:latest
   ```

5. **Create Container App Environment**
   ```bash
   az containerapp env create \
     --name free-claude-code-env \
     --resource-group free-claude-code-rg \
     --location eastus
   ```

6. **Deploy Container App**
   ```bash
   az containerapp create \
     --name free-claude-code \
     --resource-group free-claude-code-rg \
     --environment free-claude-code-env \
     --image freeclaudecode.azurecr.io/free-claude-code:latest \
     --target-port 8082 \
     --ingress external \
     --env-vars \
       NVIDIA_NIM_API_KEY=secretref:nvidia-key \
       ANTHROPIC_AUTH_TOKEN=secretref:auth-token \
     --secrets \
       nvidia-key="nvapi-xxxxx" \
       auth-token="your-token"
   ```

**Cost**: Free tier (180,000 vCPU-seconds/month), then ~$10-30/month

---

### 7. DigitalOcean App Platform

**Why DigitalOcean?** Simple pricing, great docs, good performance.

#### **Steps:**

1. **Sign up** at [DigitalOcean](https://cloud.digitalocean.com)

2. **Create App**
   - Apps → Create App
   - Source: GitHub
   - Select repository
   - Auto-detect Dockerfile

3. **Configure**
   - Name: free-claude-code
   - Region: Choose closest
   - Size: Basic ($5/month)
   - HTTP Port: 8082

4. **Environment Variables**
   - Click "Edit" next to environment variables
   - Add all required variables
   - Mark secrets as "encrypted"

5. **Deploy**
   - Click "Create Resources"
   - Wait for build to complete

**Cost**: $5-12/month

---

### 8. Self-Hosted Docker

For VPS providers like Linode, Vultr, Hetzner, or your own server.

#### **Steps:**

1. **Install Docker**
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **Clone Repository**
   ```bash
   git clone https://github.com/yourname/free-claude-code.git
   cd free-claude-code
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   nano .env  # Edit with your values
   ```

4. **Deploy with Docker Compose**
   ```bash
   docker-compose up -d
   ```

5. **Setup Reverse Proxy (Nginx)**

   Create `/etc/nginx/sites-available/free-claude-code`:
   ```nginx
   server {
       listen 80;
       server_name api.yourdomain.com;

       location / {
           proxy_pass http://localhost:8082;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           
           # SSE support
           proxy_buffering off;
           proxy_cache off;
       }
   }
   ```

   Enable site:
   ```bash
   sudo ln -s /etc/nginx/sites-available/free-claude-code /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

6. **Setup SSL with Let's Encrypt**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d api.yourdomain.com
   ```

7. **Monitor Logs**
   ```bash
   docker-compose logs -f
   ```

**Cost**: $5-20/month (VPS) + domain (~$12/year)

---

## Environment Variables Checklist

### **Required (Choose ONE provider)**

```bash
# NVIDIA NIM (Recommended - 40 req/min free)
NVIDIA_NIM_API_KEY="nvapi-xxxxx"

# OR OpenRouter (Hundreds of models)
OPENROUTER_API_KEY="sk-or-xxxxx"

# OR DeepSeek (Direct API)
DEEPSEEK_API_KEY="sk-xxxxx"
```

### **Model Configuration**

```bash
# Default model (fallback)
MODEL="nvidia_nim/moonshotai/kimi-k2-instruct"

# Per-tier routing (optional)
MODEL_OPUS="nvidia_nim/z-ai/glm-4.7"
MODEL_SONNET="nvidia_nim/moonshotai/kimi-k2-instruct"
MODEL_HAIKU="nvidia_nim/stepfun-ai/step-3.5-flash"
```

### **Security (IMPORTANT for Production!)**

```bash
# Server API key - REQUIRED for production
ANTHROPIC_AUTH_TOKEN="your-secret-token-here"

# CORS origins (comma-separated)
CORS_ORIGINS="https://your-app.com,https://api.your-app.com"
```

### **Performance Tuning**

```bash
PROVIDER_RATE_LIMIT=40          # Requests per window
PROVIDER_RATE_WINDOW=60         # Window in seconds
PROVIDER_MAX_CONCURRENCY=5      # Max parallel requests

HTTP_READ_TIMEOUT=120           # Read timeout (seconds)
HTTP_WRITE_TIMEOUT=10           # Write timeout
HTTP_CONNECT_TIMEOUT=2          # Connect timeout
```

### **Features**

```bash
ENABLE_CAVEMAN=true             # Agentic enhancements
ENABLE_THINKING=true            # Thinking token support
FALLBACK_ROUTING=true           # Opus→Sonnet→Haiku fallback
```

### **Server Configuration**

```bash
HOST="0.0.0.0"                  # Listen address
PORT=8082                       # Port (Render overrides with $PORT)
LOG_FILE="server.log"           # Log file path
```

### **Optional: Discord/Telegram Bot**

```bash
MESSAGING_PLATFORM="discord"    # or "telegram"
DISCORD_BOT_TOKEN="xxxxx"
ALLOWED_DISCORD_CHANNELS="123456789,987654321"

# OR
TELEGRAM_BOT_TOKEN="xxxxx"
ALLOWED_TELEGRAM_USER_ID="123456789"
```

---

## SSL/TLS Configuration

Most platforms provide automatic SSL certificates:

| Platform | SSL | Custom Domain |
|----------|-----|---------------|
| Render | ✅ Auto | ✅ Free |
| Railway | ✅ Auto | ✅ Free |
| Fly.io | ✅ Auto | ✅ Free |
| Cloud Run | ✅ Auto | ✅ Free |
| Azure | ✅ Auto | ✅ Paid |
| AWS | Via ALB | Via Route53 |
| Self-Hosted | Certbot | DNS required |

### **Let's Encrypt (Self-Hosted)**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d api.yourdomain.com

# Auto-renewal (cron)
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

---

## Networking & Security

### **Outbound IP Addresses by Platform**

If your API provider requires IP whitelisting, here are the outbound IP ranges:

| Platform | Outbound IPs | Whitelisting |
|----------|--------------|--------------|
| **Render** | `74.220.52.0/24`, `74.220.60.0/24` | Shared IPs |
| **Railway** | Dynamic (changes) | Not recommended |
| **Fly.io** | Region-specific (varies) | Check: `fly ips list` |
| **Cloud Run** | Dynamic from GCP ranges | Use Cloud NAT for static IP |
| **AWS Fargate** | NAT Gateway IP | Configure NAT Gateway |
| **Azure** | Virtual Network NAT | Configure NAT Gateway |
| **DigitalOcean** | Droplet IP | Static per droplet |
| **Self-Hosted** | Your VPS IP | Fully controllable |

**Important Notes:**
- Most API providers (NVIDIA NIM, OpenRouter, DeepSeek) **do not require IP whitelisting**
- Shared IPs (like Render) are not unique to your service
- For static IPs on cloud platforms, use NAT Gateways (adds cost)

### **Getting Static Outbound IP**

If you need a static outbound IP for API whitelisting:

**Google Cloud Run with Cloud NAT:**
```bash
# Create Cloud Router
gcloud compute routers create free-claude-router \
  --network=default \
  --region=us-central1

# Reserve static IP
gcloud compute addresses create free-claude-ip \
  --region=us-central1

# Create Cloud NAT
gcloud compute routers nats create free-claude-nat \
  --router=free-claude-router \
  --region=us-central1 \
  --nat-external-ip-pool=free-claude-ip \
  --nat-all-subnet-ip-ranges
```

**AWS Fargate with NAT Gateway:**
```bash
# Allocate Elastic IP
aws ec2 allocate-address --domain vpc

# Create NAT Gateway in public subnet
aws ec2 create-nat-gateway \
  --subnet-id subnet-xxxxx \
  --allocation-id eipalloc-xxxxx
```

---

## Monitoring & Observability

### **Health Check Endpoint**

All platforms should use: `GET /health`

**Expected response:**
```json
{
  "status": "healthy"
}
```

### **Logging**

**View logs by platform:**

```bash
# Render
# Dashboard → Logs tab

# Railway
railway logs

# Fly.io
fly logs

# AWS
aws logs tail /ecs/free-claude-code --follow

# Google Cloud
gcloud run logs read free-claude-code --follow

# Docker
docker-compose logs -f
```

### **Metrics to Monitor**

1. **Request Rate** - Requests per minute
2. **Error Rate** - 5xx errors / total requests
3. **P95 Latency** - 95th percentile response time
4. **Provider Rate Limits** - 429 errors count
5. **Memory Usage** - Should stay < 1GB
6. **CPU Usage** - Should stay < 80%

### **Alerting Setup**

**Recommended alerts:**

```yaml
- Error rate > 5% for 5 minutes
- P95 latency > 10 seconds
- Memory usage > 80%
- Service health check failing
- Provider rate limit exceeded
```

---

## Troubleshooting

### **Common Issues**

#### **1. "Container exits with code 1"**

**Cause:** Missing required environment variables

**Fix:** Check logs and ensure all required variables are set:
```bash
NVIDIA_NIM_API_KEY=xxx  # OR OPENROUTER_API_KEY OR DEEPSEEK_API_KEY
MODEL=nvidia_nim/moonshotai/kimi-k2-instruct
```

#### **2. "Health check failing"**

**Cause:** Wrong port configuration

**Fix:** Ensure container listens on correct port:
- Dockerfile exposes 8082
- `PORT` env var matches (Render auto-sets this)
- Health check points to `/health`

#### **3. "401 Authentication failed"**

**Cause:** API key invalid or not set

**Fix:** Verify API key is correct:
```bash
# Test NVIDIA NIM key
curl -H "Authorization: Bearer nvapi-xxxxx" https://integrate.api.nvidia.com/v1/models

# Test OpenRouter key
curl -H "Authorization: Bearer sk-or-xxxxx" https://openrouter.ai/api/v1/models
```

#### **4. "429 Rate limit exceeded"**

**Cause:** Hitting provider rate limits

**Fix:** Adjust rate limiting:
```bash
PROVIDER_RATE_LIMIT=30          # Reduce from 40
PROVIDER_MAX_CONCURRENCY=3      # Reduce from 5
```

#### **5. "CORS error in browser"**

**Cause:** CORS not configured

**Fix:** Set allowed origins:
```bash
CORS_ORIGINS="https://your-frontend.com"
```

#### **6. "Connection timeout"**

**Cause:** Read timeout too short for large requests

**Fix:** Increase timeout:
```bash
HTTP_READ_TIMEOUT=300           # 5 minutes for Opus
HTTP_READ_TIMEOUT_OPUS=600      # 10 minutes for Opus specifically
```

#### **7. "Memory limit exceeded"**

**Cause:** Container running out of memory

**Fix:** Increase memory allocation:
- Render: Upgrade to Standard plan (2GB RAM)
- Fly.io: `fly scale memory 1024`
- AWS: Increase task memory in task definition
- Docker: `docker update --memory 1g container_name`

#### **8. "Provider API blocked/403 Forbidden"**

**Cause:** Provider requires IP whitelisting

**Fix:** 
1. Check if your provider requires IP whitelisting (most don't)
2. Whitelist Render IPs in provider dashboard:
   ```
   74.220.52.0/24
   74.220.60.0/24
   ```
3. Or deploy on platform with static IP (self-hosted, dedicated NAT Gateway)

**Test provider connectivity:**
```bash
# From your Render service logs, check if requests reach provider
curl -H "Authorization: Bearer YOUR_API_KEY" https://integrate.api.nvidia.com/v1/models
```

### **Debug Mode**

Enable verbose logging:

```bash
# Set log level
LOG_LEVEL=DEBUG

# View detailed request/response logs
docker-compose logs -f | grep DEBUG
```

### **Test Deployment**

```bash
# Health check
curl https://your-app.onrender.com/health

# Test API endpoint (no auth)
curl https://your-app.onrender.com/

# Test with authentication
curl -H "x-api-key: your-token" https://your-app.onrender.com/v1/messages
```

---

## Production Checklist

Before going live, ensure:

- [ ] ✅ All CI checks pass (format, lint, type, tests)
- [ ] ✅ `ANTHROPIC_AUTH_TOKEN` is set (never deploy without auth!)
- [ ] ✅ `CORS_ORIGINS` configured (restrict to your domains)
- [ ] ✅ SSL/TLS enabled (HTTPS)
- [ ] ✅ Health check endpoint working (`/health`)
- [ ] ✅ Rate limits configured appropriately
- [ ] ✅ Monitoring and alerting setup
- [ ] ✅ Logs are accessible and structured
- [ ] ✅ Backup strategy for session data (if using bots)
- [ ] ✅ Load testing completed
- [ ] ✅ Documentation updated with your domain

---

## Support

- **Issues**: https://github.com/whitevshadow/Free-Claude-code/issues
- **Discussions**: https://github.com/whitevshadow/Free-Claude-code/discussions
- **Discord**: [Join our community](#)

---

## Cost Comparison

| Platform | Free Tier | Paid Tier | SSL | Auto-Scale |
|----------|-----------|-----------|-----|------------|
| **Render** | ✅ (sleeps) | $7/mo | ✅ | ✅ |
| **Railway** | $5 credit | ~$5-20/mo | ✅ | ✅ |
| **Fly.io** | ✅ Limited | ~$5-15/mo | ✅ | ✅ |
| **Cloud Run** | ✅ Generous | Pay-per-use | ✅ | ✅ |
| **AWS Fargate** | ❌ | ~$15-30/mo | Via ALB | ✅ |
| **Azure** | ✅ Limited | ~$10-30/mo | ✅ | ✅ |
| **DigitalOcean** | ❌ | $5-12/mo | ✅ | ❌ |
| **Self-Hosted** | - | $5-20/mo | Certbot | Manual |

**Recommendation for beginners**: Start with **Render** (easiest) or **Railway** (great DX)

**Recommendation for scale**: **Google Cloud Run** (serverless) or **AWS Fargate** (enterprise)

---

**Happy Deploying! 🚀**
