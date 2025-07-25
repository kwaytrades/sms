from loguru import logger

class StripeService:
    def __init__(self):
        logger.info("StripeService initialized (mock mode)")
    
    async def create_payment_link(self, user_id: str, plan_type: str):
        return f"https://mock-stripe-link.com/{plan_type}/{user_id}"
