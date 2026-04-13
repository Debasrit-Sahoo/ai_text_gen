from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from core.inference import run_inference, inference_lock
import asyncio

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

    @field_validator('message')
    @classmethod
    def message_length(cls, v):
        if len(v) > 4000:
            raise ValueError('message too long')
        return v

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    async with inference_lock:
        try:
            loop = asyncio.get_event_loop()
            reply = await asyncio.wait_for(
                loop.run_in_executor(None, run_inference, req.message),
                timeout=300
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="inference timed out")
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
    return ChatResponse(response=reply)