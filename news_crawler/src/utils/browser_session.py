# utils/browser_session.py

import random
from typing import Dict


class BrowserSession:
    """Manages browser-like session with realistic headers and behavior"""

    USER_AGENTS = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        # Chrome on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
        # Firefox on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/120.0",
        # Safari on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
        # Edge on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    ]

    def __init__(self):
        self.user_agent = random.choice(self.USER_AGENTS)
        self.session_id = self._generate_session_id()

    def _generate_session_id(self) -> str:
        """Generate realistic session ID"""
        import uuid

        return str(uuid.uuid4())

    def get_headers(self, is_graphql: bool = True) -> Dict[str, str]:
        """Get realistic browser headers"""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": (
                "application/json, text/plain, */*"
                if is_graphql
                else "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "empty" if is_graphql else "document",
            "Sec-Fetch-Mode": "cors" if is_graphql else "navigate",
            "Sec-Fetch-Site": "same-origin" if is_graphql else "none",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

        if is_graphql:
            headers.update(
                {
                    "Content-Type": "application/json",
                    "Origin": "https://cointelegraph.com",
                    "Referer": "https://cointelegraph.com/",
                    "X-Requested-With": "XMLHttpRequest",
                }
            )

        return headers
