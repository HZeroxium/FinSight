# utils/browser_session.py

import random
import uuid
from typing import Dict, List


class BrowserSession:
    """Manages browser-like session with realistic headers and behavior"""

    # Expanded list of realistic User-Agent strings
    USER_AGENTS: List[str] = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        # Edge on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0",
        # Opera on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) OPR/90.0.0.0 Safari/537.36",
        # Chrome on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        # Safari on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
        # Chrome on Linux
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        # Firefox on Linux
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
        # Chrome on Android
        "Mozilla/5.0 (Linux; Android 14; Pixel 7 Build/T4B1.240822.004) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6097.94 Mobile Safari/537.36",
        # Firefox on Android
        "Mozilla/5.0 (Android 14; Mobile; rv:120.0) Gecko/120.0 Firefox/120.0",
        # Opera on Android
        "Mozilla/5.0 (Linux; Android 14; SM-G991B Build/RP1A.240720.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36 OPR/88.0.0.0",
        # Safari on iPhone
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/605.1.15",
        # Safari on iPad
        "Mozilla/5.0 (iPad; CPU OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/605.1.15",
        # UC Browser on Android
        "Mozilla/5.0 (Linux; U; Android 11; en-US; SM-A515F) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 UCBrowser/14.3.8.1027 U3/0.8.0 Mobile Safari/537.36",
        # Samsung Internet on Android
        "Mozilla/5.0 (Linux; Android 14; SAMSUNG SM-S908E Build/QP1A.190711.020; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/120.0.0.0 Mobile Safari/537.36 SamsungBrowser/20.0",
        # Internet Explorer 11
        "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko",
    ]

    # Common Accept-Language headers
    ACCEPT_LANGUAGES: List[str] = [
        "en-US,en;q=0.9",
        "en-GB,en;q=0.8",
        "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "es-ES,es;q=0.9,en-US;q=0.8,en;q=0.7",
    ]

    # Common Accept-Encoding headers
    ACCEPT_ENCODINGS: List[str] = [
        "gzip, deflate, br",
        "gzip, deflate",
        "br",
        "gzip",
        "deflate",
    ]

    def __init__(self):
        # Pick a random User-Agent for the session
        self.user_agent: str = random.choice(self.USER_AGENTS)
        # Generate a unique session ID for cookie header
        self.session_id: str = self._generate_session_id()

    def _generate_session_id(self) -> str:
        """Generate a realistic session ID using UUID4"""
        return str(uuid.uuid4())

    def get_headers(
        self,
        is_graphql: bool = True,
        origin: str = None,
        referer: str = None,
    ) -> Dict[str, str]:
        """Get realistic browser headers for requests"""
        headers: Dict[str, str] = {
            "User-Agent": self.user_agent,
            "Accept": (
                "application/json, text/plain, */*"
                if is_graphql
                else "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
            ),
            # Randomize language and encoding for realism
            "Accept-Language": random.choice(self.ACCEPT_LANGUAGES),
            "Accept-Encoding": random.choice(self.ACCEPT_ENCODINGS),
            # Common security and control headers
            "DNT": "1",
            "Connection": random.choice(["keep-alive", "close"]),
            "Upgrade-Insecure-Requests": "1",
            # Fetch metadata
            "Sec-Fetch-Dest": "empty" if is_graphql else "document",
            "Sec-Fetch-Mode": "cors" if is_graphql else "navigate",
            "Sec-Fetch-Site": "same-origin" if is_graphql else "none",
            # Force fresh responses
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            # Include session cookie for consistency
            "Cookie": f"session_id={self.session_id}",
        }

        if is_graphql:
            headers.update(
                {
                    "Content-Type": "application/json",
                    # Use provided or default origin/referer
                    "Origin": origin or "https://cointelegraph.com",
                    "Referer": referer or "https://cointelegraph.com/",
                    "X-Requested-With": "XMLHttpRequest",
                }
            )

        return headers
