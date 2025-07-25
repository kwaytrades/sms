# sms

# ===== README.md =====
# SMS Trading Bot

Hyper-personalized SMS trading insights with behavioral learning.

## Features

- 🤖 AI-powered personalized trading insights
- 📱 SMS-based interface with natural conversations
- 📊 Real-time portfolio analysis via Plaid
- 🧠 Behavioral learning that adapts to user patterns
- 💰 Subscription management with Stripe
- 📈 Daily premarket and market close insights
- 🔍 Intelligent stock screener

## Pricing

- **FREE**: 10 messages/week
- **PAID**: $29/month - 100 messages + personalized insights
- **PRO**: $99/month - Unlimited + trade alerts

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in API keys
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python main.py`

## API Endpoints

- `POST /webhook/sms` - Twilio SMS webhook
- `POST /webhook/stripe` - Stripe payment webhook  
- `GET /health` - Health check
- `GET /admin` - Admin dashboard

## Architecture

```
SMS → FastAPI → MongoDB (conversations) + Redis (cache) → OpenAI → Response
```

## Deployment

Deploy to Render with the included Dockerfile and environment variables.
