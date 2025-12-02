"""
Perplexity API client with automatic ARC Saga integration
"""

import json
import uuid
from typing import List, Dict, Optional, AsyncIterator
from datetime import datetime
from openai import AsyncOpenAI


class PerplexityClient:
    def __init__(self, api_key: str, storage):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        self.storage = storage

    async def ask_streaming(
        self,
        query: str,
        context: List[Dict] = None,
        thread_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Ask Perplexity with streaming response
        Automatically stores conversation in ARC Saga
        """

        if not thread_id:
            thread_id = str(uuid.uuid4())

        # Build messages
        messages = []

        # Add context if provided
        if context:
            messages.extend(context)

        # Add current query
        messages.append({"role": "user", "content": query})

        # Store user message
        from ..models import Message, MessageRole
        user_msg = Message(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            role=MessageRole.USER,
            content=query,
            provider="perplexity",
            timestamp=datetime.now()
        )
        self.storage.store_message(user_msg)

        # Call Perplexity API
        full_response = ""

        try:
            stream = await self.client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content

                    # Yield chunk to client
                    yield json.dumps({
                        "type": "chunk",
                        "content": content
                    })

            # Store complete assistant response
            assistant_msg = Message(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                role=MessageRole.ASSISTANT,
                content=full_response,
                provider="perplexity",
                timestamp=datetime.now()
            )
            self.storage.store_message(assistant_msg)

            # Send completion signal
            yield json.dumps({
                "type": "complete",
                "thread_id": thread_id
            })

        except Exception as e:
            yield json.dumps({
                "type": "error",
                "message": str(e)
            })
