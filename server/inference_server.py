"""
FastAPI Inference Server для TRM моделі
OpenAI-compatible REST API
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
from pathlib import Path
import json

app = FastAPI(title="TRM Inference Server")

# Глобальні змінні для моделі та tokenizer
model = None
tokenizer = None
device = "cpu"


class CompletionRequest(BaseModel):
    """Request для генерації тексту"""
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None


class CompletionResponse(BaseModel):
    """Response з згенерованим текстом"""
    text: str
    tokens_generated: int


@app.on_event("startup")
def load():
    """Завантажити модель та tokenizer на startup"""
    global model, tokenizer, device
    
    # Шукати модель
    model_path = Path("models/trained")
    if not model_path.exists():
        model_path = Path("checkpoints")
    
    # Знайти останню модель
    model_files = list(model_path.glob("*.pt"))
    if not model_files:
        print("⚠️ Модель не знайдено, inference буде недоступний")
        return
    
    # Завантажити останню модель
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"📦 Завантаження моделі: {latest_model}")
    
    try:
        checkpoint = torch.load(latest_model, map_location=device)
        if 'model_state_dict' in checkpoint:
            # Потрібна конфігурація для створення моделі
            # Тимчасово використовуємо дефолти
            from train.model_factory import create_model
            model = create_model(
                dim=256,
                vocab_size=50257,
                depth=4,
                seq_len=256
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Модель вже завантажена
            model = checkpoint
        
        model.eval()
        model.to(device)
        print("✅ Модель завантажена")
    except Exception as e:
        print(f"❌ Помилка завантаження моделі: {e}")
        model = None
    
    # Завантажити tokenizer
    try:
        from tiny_recursive_model.utils import load_tokenizer
        tokenizer, _, _ = load_tokenizer("gpt2")
        print("✅ Tokenizer завантажено")
    except Exception as e:
        print(f"❌ Помилка завантаження tokenizer: {e}")
        tokenizer = None


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "TRM Inference Server",
        "status": "running",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }


@app.post("/v1/completions", response_model=CompletionResponse)
def complete(req: CompletionRequest):
    """
    Генерація тексту (OpenAI-compatible)
    
    Args:
        req: Request з prompt та параметрами
    
    Returns:
        Response з згенерованим текстом
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Модель або tokenizer не завантажені")
    
    try:
        # Encode prompt
        tokens = tokenizer.encode(req.prompt)
        tokens_tensor = torch.tensor([tokens], device=device)
        
        # Генерація (спрощена версія)
        model.eval()
        with torch.no_grad():
            # Тимчасово використовуємо просту генерацію
            # В реальності потрібна повна генерація з sampling
            output = model(tokens_tensor)
            generated_tokens = output.argmax(dim=-1)[0].cpu().tolist()
        
        # Decode
        generated_text = tokenizer.decode(generated_tokens[:req.max_tokens])
        tokens_generated = len(generated_tokens[:req.max_tokens])
        
        return CompletionResponse(
            text=generated_text,
            tokens_generated=tokens_generated
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка генерації: {str(e)}")


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

