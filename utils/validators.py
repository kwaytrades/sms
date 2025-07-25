import re
from typing import Optional

def validate_phone_number(phone: str) -> str:
    """Validate and format phone number"""
    # Remove all non-digits
    digits = re.sub(r'\D', '', phone)
    
    # Add country code if missing
    if len(digits) == 10:
        digits = "1" + digits
    
    # Validate length
    if len(digits) != 11 or not digits.startswith('1'):
        raise ValueError(f"Invalid phone number: {phone}")
    
    return f"+{digits}"

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def sanitize_input(text: str) -> str:
    """Sanitize user input"""
    # Remove potential harmful characters
    text = re.sub(r'[<>\"\'&]', '', text)
    return text.strip()[:1000]  # Limit length
