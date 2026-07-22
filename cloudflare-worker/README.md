# Cloudflare Worker — Telegram Relay

Forwards alerts from Unbreakable Eye to Telegram API.

## Deploy

```bash
cd cloudflare-worker

# Install wrangler (if not installed)
npm install -g wrangler

# Login to Cloudflare
npx wrangler login

# Deploy the worker
npx wrangler deploy

# Set secrets
npx wrangler secret put TELEGRAM_BOT_TOKEN   # Paste your bot token
npx wrangler secret put ALLOWED_TOKENS        # Paste a secret token (for auth)
npx wrangler secret put DEFAULT_CHAT_ID       # Paste your chat ID
```

## Get Your Worker URL

After deployment, wrangler prints your worker URL:
```
https://unbreakable-eye-telegram.<your-subdomain>.workers.dev
```

## Update .env

Add to your `.env`:
```
CLOUDFLARE_WORKER_URL=https://unbreakable-eye-telegram.<your-subdomain>.workers.dev
WORKER_SECRET=the-same-token-you-set-as-ALLOWED_TOKENS
```

## Test

```bash
curl -X POST https://your-worker.workers.dev/send \
  -H "Content-Type: application/json" \
  -d '{"message":"Test alert","token":"your-secret"}'
```

## Rebuild Docker

```bash
docker compose build fastapi && docker compose up -d
```
