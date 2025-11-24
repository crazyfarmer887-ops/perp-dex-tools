import os
import aiohttp
from typing import Dict, Any, Optional
import certifi
import ssl

BASE_URL = "https://api.telegram.org/bot"

class TelegramBot:
    def __init__(self, token: str, chat_id: str, base_url: Optional[str] = None):
        self.token = token
        self.chat_id = chat_id
        self.base_url = base_url if base_url else BASE_URL
        self.api_url = f"{self.base_url.rstrip('/')}{self.token}"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=ssl.create_default_context(cafile=certifi.where()))
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """close requests session"""
        if self.session:
            await self.session.close()

    async def send_text(self, content: str, parse_mode: str = "HTML") -> Dict[str, Any]:
        """Send a text message to Telegram"""
        payload = {
            "chat_id": self.chat_id,
            "text": content,
            "parse_mode": parse_mode
        }
        return await self._send_message("sendMessage", payload)

    async def _send_message(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to send messages to Telegram API"""
        url = f"{self.api_url}/{method}"
        
        if self.session is None:
             self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=ssl.create_default_context(cafile=certifi.where()))
            )

        try:
            async with self.session.post(url, json=payload) as response:
                response_data = await response.json()
                if not response_data.get("ok", False):
                    print(f"Telegram send message failed: {response_data}")
                return response_data
        except Exception as e:
            print(f"Telegram send message failed: {e}")
            return {"ok": False, "error": str(e)}
