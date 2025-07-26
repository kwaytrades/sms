# 🚀 SMS Trading Bot - Unified System

A complete SMS-based AI trading assistant with built-in technical analysis, user management, and subscription billing.

## ✨ Features

- **🤖 AI-Powered Trading Insights** - Personalized analysis using OpenAI
- **📱 SMS Interface** - Natural language conversations via Twilio
- **📊 Technical Analysis** - Built-in RSI, MACD, Support/Resistance, Gap Analysis
- **👥 User Management** - Behavioral learning and personalization
- **💰 Subscription Plans** - Free, Paid ($29/mo), Pro ($99/mo) via Stripe
- **⚡ Smart Caching** - Redis-based caching with market-aware TTL
- **📈 Real-time Data** - EODHD market data integration
- **🔒 Security** - Proper input validation and API key management

## 🏗️ Architecture

```
SMS User ←→ Twilio ←→ FastAPI App ←→ MongoDB (Users/Conversations)
                           ↓
                      Technical Analysis ←→ Redis Cache
                           ↓
                      EODHD API (Market Data)
                           ↓
                      OpenAI (Personalized Responses)
```

## 📋 Prerequisites

- Python 3.11+
- MongoDB Atlas account
- Redis instance (or Upstash)
- API Keys for:
  - OpenAI (for AI responses)
  - EODHD (for market data)
  - Twilio (for SMS)
  - Stripe (for payments)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd sms-trading-bot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your API keys and database URLs
```

### 3. Required API Keys

**MongoDB Atlas:**
```bash
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/sms_trading_bot
```

**Redis (Upstash or local):**
```bash
REDIS_URL=redis://localhost:6379
# Or for Upstash: rediss://username:password@host:port
```

**OpenAI:**
```bash
OPENAI_API_KEY=sk-your-key-here
```

**EODHD (Market Data):**
```bash
EODHD_API_KEY=your-eodhd-key
```

**Twilio (SMS):**
```bash
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1234567890
```

**Stripe (Payments):**
```bash
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PAID_PRICE_ID=price_...
STRIPE_PRO_PRICE_ID=price_...
```

### 4. Run Locally

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## 📱 SMS Commands

Users can text any of these commands:

### Basic Commands
- `START` - Welcome message and setup
- `HELP` - Show all available commands
- `STATUS` - Account overview and usage
- `UPGRADE` - View subscription plans

### Trading Commands
- `How is AAPL?` - Get stock analysis
- `AAPL price` - Get current price
- `Find me tech stocks` - Stock screener (coming soon)

### Account Management
- `WATCHLIST` - Manage favorite stocks
- `CANCEL` - Cancel subscription
- `STOP` - Unsubscribe from messages

## 🎯 Subscription Plans

| Plan | Price | Weekly Limit | Features |
|------|-------|--------------|----------|
| **Free** | $0 | 4 messages | Basic stock analysis |
| **Paid** | $29/mo | 40 messages | Personalized insights, portfolio tracking |
| **Pro** | $99/mo | 120 messages | Unlimited daily (50/day cooloff), real-time alerts |

## 📊 API Endpoints

### Core Endpoints
- `POST /webhook/sms` - Twilio SMS webhook
- `POST /webhook/stripe` - Stripe payment webhook
- `GET /health` - Health check
- `GET /` - API information

### Technical Analysis
- `GET /analysis/{symbol}` - Complete technical analysis
- `GET /signals/{symbol}` - Trading signals only
- `GET /cache/stats` - Cache statistics
- `POST /cache/clear` - Clear cache
- `DELETE /cache/{symbol}` - Invalidate symbol cache

### Admin Dashboard
- `GET /admin` - Admin dashboard
- `GET /admin/users/{phone}` - User profile
- `POST /admin/users/{phone}/subscription` - Update subscription

## 🛠️ Development

### Local Development with Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Testing SMS Webhooks

Use ngrok to expose your local server:

```bash
# Install ngrok
npm install -g ngrok

# Expose port 8000
ngrok http 8000

# Update Twilio webhook URL to: https://your-ngrok-url.ngrok.io/webhook/sms
```

### Database Setup

The app automatically creates indexes on startup. For manual setup:

```javascript
// MongoDB indexes
db.users.createIndex({ "phone_number": 1 }, { unique: true })
db.users.createIndex({ "plan_type": 1, "subscription_status": 1 })
db.conversations.createIndex({ "phone_number": 1, "timestamp": -1 })
db.usage_tracking.createIndex({ "phone_number": 1, "date": 1 })
```

## 🚀 Deployment

### Deploy to Render

1. **Fork this repository**
2. **Connect to Render:**
   - Go to [render.com](https://render.com)
   - Create new Web Service
   - Connect your GitHub repo
   - Use these settings:
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Set Environment Variables in Render:**
   ```
   ENVIRONMENT=production
   MONGODB_URL=your-mongodb-url
   REDIS_URL=your-redis-url
   OPENAI_API_KEY=your-openai-key
   EODHD_API_KEY=your-eodhd-key
   TWILIO_ACCOUNT_SID=your-twilio-sid
   TWILIO_AUTH_TOKEN=your-twilio-token
   TWILIO_PHONE_NUMBER=your-twilio-number
   STRIPE_SECRET_KEY=your-stripe-key
   STRIPE_WEBHOOK_SECRET=your-webhook-secret
   STRIPE_PAID_PRICE_ID=your-paid-price-id
   STRIPE_PRO_PRICE_ID=your-pro-price-id
   ```

4. **Configure Webhooks:**
   - **Twilio:** Set webhook URL to `https://your-app.onrender.com/webhook/sms`
   - **Stripe:** Set webhook URL to `https://your-app.onrender.com/webhook/stripe`

### Deploy to Other Platforms

The app works on any platform supporting Python and Docker:

- **Railway:** Connect GitHub repo, set environment variables
- **Fly.io:** `fly deploy` with Dockerfile
- **Google Cloud Run:** Deploy from source
- **AWS ECS:** Use provided Dockerfile

## 🔧 Configuration

### Cache Settings

```python
# config.py
CACHE_POPULAR_TTL = 1800  # 30 min for popular stocks (AAPL, TSLA, etc.)
CACHE_ONDEMAND_TTL = 300   # 5 min for other stocks
CACHE_AFTERHOURS_TTL = 3600 # 1 hour when market closed
```

### Plan Limits

```python
# config.py
PLAN_LIMITS = {
    "free": {"weekly_limit": 4, "price": 0},
    "paid": {"weekly_limit": 40, "price": 29}, 
    "pro": {"weekly_limit": 120, "price": 99}
}
```

### Popular Tickers (Enhanced Caching)

Edit `POPULAR_TICKERS` in `config.py` to customize which stocks get longer cache TTL:

```python
POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
    # Add your most requested stocks here
]
```

## 📚 Key Features Explained

### 🧠 Behavioral Learning

The system learns from each user interaction:

- **Communication style** (formal/casual, emoji usage)
- **Technical depth** preference (basic/advanced)
- **Common tickers** they ask about
- **Question patterns** and response timing

### ⚡ Smart Caching

Three-tier caching strategy:

1. **Popular stocks** (AAPL, TSLA, etc.) → 30-minute cache during market hours
2. **Regular stocks** → 5-minute cache during market hours  
3. **After hours** → 1-hour cache for all stocks

### 💬 Natural Language Processing

Users can ask naturally:
- "How is Apple doing?" → Analyzes AAPL
- "What's Tesla's RSI?" → Returns RSI for TSLA
- "Find me cheap tech stocks" → Runs stock screener

### 📊 Technical Analysis

Built-in calculations for:
- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)
- **EMA** (Exponential Moving Averages)
- **Bollinger Bands**
- **VWAP** (Volume Weighted Average Price)
- **ATR** (Average True Range)
- **Support/Resistance** levels
- **Price Gap** detection

## 🔒 Security Features

### Input Validation
- Phone number format validation
- SQL injection prevention
- XSS protection
- Rate limiting

### API Security
- **No hardcoded API keys** (fixed security issue)
- Environment variable management
- Webhook signature verification
- Secure database connections

### Data Protection
- User data encryption at rest
- Secure SMS transmission
- PCI compliance via Stripe
- GDPR-friendly data handling

## 🐛 Troubleshooting

### Common Issues

**1. SMS not sending:**
```bash
# Check Twilio credentials
curl -X GET "https://api.twilio.com/2010-04-01/Accounts.json" \
  -u your_account_sid:your_auth_token
```

**2. Database connection failed:**
```bash
# Test MongoDB connection
python -c "from pymongo import MongoClient; print(MongoClient('your-mongodb-url').admin.command('ping'))"
```

**3. Cache not working:**
```bash
# Test Redis connection
redis-cli -u your-redis-url ping
```

**4. Technical analysis errors:**
```bash
# Check EODHD API
curl "https://eodhd.com/api/eod/AAPL.US?api_token=your-api-key&fmt=json"
```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
uvicorn main:app --reload
```

### Health Checks

Monitor service health:

```bash
# Local
curl http://localhost:8000/health

# Production  
curl https://your-app.onrender.com/health
```

## 📈 Monitoring & Analytics

### Built-in Metrics

Access at `/admin`:
- Total users by plan
- Message volume
- Cache hit rates
- Popular stocks requested
- Error rates

### Log Analysis

Key log patterns to monitor:

```bash
# Successful SMS processing
grep "Successfully processed message" logs/

# Cache performance
grep "Cache HIT\|Cache MISS" logs/

# API errors
grep "ERROR" logs/

# Subscription changes
grep "Updated subscription" logs/
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

### Code Style

- Use `black` for code formatting
- Follow PEP 8 guidelines
- Add type hints for new functions
- Include docstrings for public methods

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation:** This README
- **Issues:** GitHub Issues
- **Email:** support@yourdomain.com

## 🎯 Roadmap

### Phase 1 (Current)
- ✅ SMS interface with natural language
- ✅ Technical analysis with caching
- ✅ User management and subscriptions
- ✅ Behavioral learning

### Phase 2 (Next)
- 🔄 Advanced stock screener
- 🔄 Portfolio tracking via Plaid
- 🔄 Real-time price alerts
- 🔄 Options analysis

### Phase 3 (Future)
- 📅 Earnings calendar integration
- 📅 News sentiment analysis
- 📅 Social trading features
- 📅 Mobile app companion

---

## 🚀 Quick Deploy Checklist

Before going live:

- [ ] All API keys configured in production environment
- [ ] Twilio webhook URL updated to production
- [ ] Stripe webhook URL updated to production  
- [ ] MongoDB indexes created
- [ ] Redis connection tested
- [ ] Health check endpoint responding
- [ ] Test SMS flow end-to-end
- [ ] Monitor logs for errors
- [ ] Set up monitoring/alerting

**Your unified SMS Trading Bot is ready to serve users! 🎉**
