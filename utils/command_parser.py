# ===== utils/command_parser.py =====
import re
from typing import Optional, Dict, List

class CommandParser:
    """Parse and validate SMS commands"""
    
    # Command patterns
    COMMANDS = {
        'start': ['start', 'begin', 'hello', 'hi'],
        'upgrade': ['/upgrade', 'upgrade', 'plans'],
        'downgrade': ['/downgrade', 'downgrade'],
        'cancel': ['/cancel', 'cancel', 'unsubscribe'],
        'billing': ['/billing', 'billing', 'payment'],
        'status': ['/status', 'status', 'account'],
        'help': ['/help', 'help', '?'],
        'watchlist': ['/watchlist', 'watchlist', 'stocks'],
        'portfolio': ['/portfolio', 'portfolio'],
        'screen': ['/screen', 'screen', 'find'],
        'alerts': ['/alerts', 'alerts', 'notifications'],
        'settings': ['/settings', 'settings', 'preferences'],
        'support': ['/support', 'support', 'help'],
        'pause': ['/pause', 'pause', 'stop'],
        'resume': ['/resume', 'resume', 'continue'],
        'stop': ['stop', 'unsubscribe', 'opt out']
    }
    
    @classmethod
    def parse_command(cls, message: str) -> Optional[str]:
        """Parse message and return command if found"""
        message = message.strip().lower()
        
        # Direct command match
        for command, patterns in cls.COMMANDS.items():
            if message in patterns:
                return command
        
        return None
    
    @classmethod
    def parse_watchlist_action(cls, message: str) -> Optional[Dict]:
        """Parse watchlist add/remove commands"""
        message = message.strip().upper()
        
        # ADD commands
        add_patterns = [
            r'^ADD\s+([A-Z]{1,5})$',
            r'^WATCH\s+([A-Z]{1,5})$',
            r'^\+\s*([A-Z]{1,5})$'
        ]
        
        for pattern in add_patterns:
            match = re.match(pattern, message)
            if match:
                return {'action': 'add', 'symbol': match.group(1)}
        
        # REMOVE commands  
        remove_patterns = [
            r'^REMOVE\s+([A-Z]{1,5})$',
            r'^DELETE\s+([A-Z]{1,5})$',
            r'^-\s*([A-Z]{1,5})$'
        ]
        
        for pattern in remove_patterns:
            match = re.match(pattern, message)
            if match:
                return {'action': 'remove', 'symbol': match.group(1)}
        
        return None
    
    @classmethod
    def parse_alert_command(cls, message: str) -> Optional[Dict]:
        """Parse alert creation commands"""
        message = message.strip().upper()
        
        # ALERT AAPL 190
        alert_pattern = r'^ALERT\s+([A-Z]{1,5})\s+(\d+(?:\.\d+)?)$'
        match = re.match(alert_pattern, message)
        
        if match:
            return {
                'action': 'create_alert',
                'symbol': match.group(1),
                'price': float(match.group(2))
            }
        
        # REMOVE alert 1
        remove_pattern = r'^REMOVE\s+ALERT\s+(\d+)$'
        match = re.match(remove_pattern, message)
        
        if match:
            return {
                'action': 'remove_alert',
                'alert_id': int(match.group(1))
            }
        
        return None
