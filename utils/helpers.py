import uuid
import re
from datetime import datetime
from typing import Optional, List

def extract_command(message: str) -> Optional[str]:
    """Extract command from message"""
    message = message.strip().lower()
    
    # Check for explicit commands
    if message.startswith('/'):
        return message.split()[0]
    
    # Check for keyword commands
    commands = {
        'start': 'start',
        'stop': 'stop', 
        'help': '/help',
        'upgrade': '/upgrade',
        'cancel': '/cancel',
        'status': '/status'
    }
    
    return commands.get(message)

def generate_session_id(user_id: str) -> str:
    """Generate unique session ID"""
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H')
    return f"{user_id}_{timestamp}_{str(uuid.uuid4())[:8]}"

def format_currency(amount: float) -> str:
    """Format currency amount"""
    if amount >= 0:
        return f"+${amount:,.2f}"
    else:
        return f"-${abs(amount):,.2f}"

def extract_stock_symbols(text: str) -> List[str]:
    """Extract stock symbols from text"""
    # Find 1-5 letter uppercase words (potential symbols)
    symbols = re.findall(r'\b[A-Z]{1,5}\b', text.upper())
    
    # Filter out common words that aren't symbols
    common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'BUT', 'DO', 'GET', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
    
    return [s for s in symbols if s not in common_words and len(s) >= 2]
