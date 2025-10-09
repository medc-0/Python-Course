"""
Meng AI Production Server
========================

Production-ready server for Meng AI with:
1. FastAPI web server
2. RESTful API endpoints
3. Model serving and inference
4. Request/response validation
5. Rate limiting and security
6. Health checks and monitoring
7. Docker containerization
8. Load balancing support

Author: Meng AI Development Team
"""

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import uvicorn
import asyncio
import time
import json
import os
import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import hashlib
import secrets
from collections import defaultdict, deque
import psutil
import gc
from dataclasses import dataclass
import yaml
from pathlib import Path

# Import Meng AI components
from meng_ai_advanced import MengAIAdvanced, MengAIConfig, ModelSize
from simple_tokenizer import SimpleTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting
class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests outside the window
        while client_requests and client_requests[0] <= now - self.window_seconds:
            client_requests.popleft()
        
        # Check if under limit
        if len(client_requests) >= self.max_requests:
            return False
        
        # Add current request
        client_requests.append(now)
        return True

# Global rate limiter
rate_limiter = RateLimiter(max_requests=1000, window_seconds=3600)

# Request/Response Models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="Input prompt for text generation")
    max_length: int = Field(default=100, ge=1, le=1000, description="Maximum length of generated text")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k sampling parameter")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling parameter")
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0, description="Repetition penalty")
    do_sample: bool = Field(default=True, description="Whether to use sampling or greedy decoding")
    num_return_sequences: int = Field(default=1, ge=1, le=5, description="Number of sequences to generate")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()

class GenerationResponse(BaseModel):
    generated_text: str = Field(..., description="Generated text")
    prompt: str = Field(..., description="Original prompt")
    generation_time: float = Field(..., description="Generation time in seconds")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    model_info: Dict[str, Any] = Field(..., description="Model information")

class BatchGenerationRequest(BaseModel):
    prompts: List[str] = Field(..., min_items=1, max_items=10, description="List of prompts")
    max_length: int = Field(default=100, ge=1, le=1000)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=1, le=100)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0)
    do_sample: bool = Field(default=True)

class BatchGenerationResponse(BaseModel):
    results: List[GenerationResponse] = Field(..., description="List of generation results")
    total_time: float = Field(..., description="Total processing time")
    batch_size: int = Field(..., description="Number of prompts processed")

class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Model name")
    model_size: str = Field(..., description="Model size category")
    vocab_size: int = Field(..., description="Vocabulary size")
    max_seq_length: int = Field(..., description="Maximum sequence length")
    parameters: int = Field(..., description="Number of parameters")
    model_size_mb: float = Field(..., description="Model size in MB")
    device: str = Field(..., description="Device used for inference")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Current timestamp")
    uptime: float = Field(..., description="Server uptime in seconds")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")
    gpu_usage: Optional[Dict[str, float]] = Field(None, description="GPU usage statistics")

class ServerConfig:
    """Server configuration"""
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'model': {
                'size': 'medium',
                'model_path': None,
                'tokenizer_path': None,
                'device': 'auto'
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 1,
                'max_request_size': 10485760,  # 10MB
                'timeout': 30
            },
            'security': {
                'api_key': None,
                'cors_origins': ['*'],
                'trusted_hosts': ['*']
            },
            'rate_limiting': {
                'max_requests': 1000,
                'window_seconds': 3600
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                # Merge with defaults
                for key, value in file_config.items():
                    if key in default_config:
                        default_config[key].update(value)
        
        return default_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

# Global server state
server_config = ServerConfig()
model = None
tokenizer = None
server_start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global model, tokenizer
    
    # Startup
    logger.info("Starting Meng AI Server...")
    
    try:
        # Load model
        model_size = ModelSize(server_config.get('model.size', 'medium'))
        model_path = server_config.get('model.model_path')
        tokenizer_path = server_config.get('model.tokenizer_path')
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            # Load from checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            config = MengAIConfig(**checkpoint.get('config', {}))
            model = MengAIAdvanced(config)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.info(f"Creating new {model_size.value} model")
            model = create_meng_ai_model(model_size, vocab_size=50000)
        
        # Load tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = SimpleTokenizer()
            tokenizer.load(tokenizer_path)
        else:
            logger.info("Creating new tokenizer")
            tokenizer = SimpleTokenizer(vocab_size=50000, min_frequency=2)
            # Build vocabulary with sample data
            from simple_tokenizer import create_sample_dataset
            sample_texts = create_sample_dataset()
            tokenizer.build_vocabulary(sample_texts)
        
        # Move model to device
        device = server_config.get('model.device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Meng AI Server...")
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

# Create FastAPI app
app = FastAPI(
    title="Meng AI Server",
    description="Production-ready API for Meng AI language model",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=server_config.get('security.cors_origins', ['*']),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=server_config.get('security.trusted_hosts', ['*'])
)

# Dependency functions
def get_client_id(request) -> str:
    """Get client identifier for rate limiting"""
    # Use IP address as client ID (in production, use API key or user ID)
    return request.client.host

def check_rate_limit(client_id: str = Depends(get_client_id)):
    """Check rate limiting"""
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return client_id

def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify API key if configured"""
    api_key = server_config.get('security.api_key')
    if api_key is None:
        return True  # No API key required
    
    if credentials is None:
        raise HTTPException(status_code=401, detail="API key required")
    
    if credentials.credentials != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Meng AI Server",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_usage = {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "percent": memory.percent
    }
    
    # Get GPU usage if available
    gpu_usage = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_usage = {
            "total_gb": gpu_memory,
            "allocated_gb": gpu_allocated,
            "available_gb": gpu_memory - gpu_allocated,
            "percent": (gpu_allocated / gpu_memory) * 100
        }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        uptime=time.time() - server_start_time,
        model_loaded=model is not None,
        memory_usage=memory_usage,
        gpu_usage=gpu_usage
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_info = model.get_model_size()
    
    return ModelInfo(
        model_name="Meng AI",
        model_size=model.config.__class__.__name__,
        vocab_size=model.vocab_size,
        max_seq_length=model.max_seq_length,
        parameters=model_info['total_parameters'],
        model_size_mb=model_info['model_size_mb'],
        device=str(next(model.parameters()).device)
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    client_id: str = Depends(check_rate_limit),
    api_key_verified: bool = Depends(verify_api_key)
):
    """Generate text from a prompt"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Encode prompt
        input_ids = tokenizer.encode(request.prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(next(model.parameters()).device)
        
        # Generate text
        with torch.no_grad():
            generated = model.generate(
                input_tensor,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(generated[0].cpu().tolist())
        
        # Remove the original prompt from generated text
        if generated_text.startswith(request.prompt):
            generated_text = generated_text[len(request.prompt):].strip()
        
        generation_time = time.time() - start_time
        tokens_generated = len(generated[0]) - len(input_ids)
        
        # Model info
        model_info = {
            "model_name": "Meng AI",
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(next(model.parameters()).device)
        }
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            generation_time=generation_time,
            tokens_generated=tokens_generated,
            model_info=model_info
        )
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/batch", response_model=BatchGenerationResponse)
async def generate_batch(
    request: BatchGenerationRequest,
    client_id: str = Depends(check_rate_limit),
    api_key_verified: bool = Depends(verify_api_key)
):
    """Generate text from multiple prompts"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        results = []
        
        for prompt in request.prompts:
            # Encode prompt
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(next(model.parameters()).device)
            
            # Generate text
            with torch.no_grad():
                generated = model.generate(
                    input_tensor,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample
                )
            
            # Decode generated text
            generated_text = tokenizer.decode(generated[0].cpu().tolist())
            
            # Remove the original prompt from generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            tokens_generated = len(generated[0]) - len(input_ids)
            
            # Model info
            model_info = {
                "model_name": "Meng AI",
                "parameters": sum(p.numel() for p in model.parameters()),
                "device": str(next(model.parameters()).device)
            }
            
            results.append(GenerationResponse(
                generated_text=generated_text,
                prompt=prompt,
                generation_time=0.0,  # Will be calculated below
                tokens_generated=tokens_generated,
                model_info=model_info
            ))
        
        total_time = time.time() - start_time
        
        # Update generation times
        for result in results:
            result.generation_time = total_time / len(results)
        
        return BatchGenerationResponse(
            results=results,
            total_time=total_time,
            batch_size=len(request.prompts)
        )
    
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

@app.post("/model/reload")
async def reload_model(
    background_tasks: BackgroundTasks,
    api_key_verified: bool = Depends(verify_api_key)
):
    """Reload the model (admin endpoint)"""
    global model, tokenizer
    
    def reload():
        global model, tokenizer
        try:
            # Unload current model
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            # Reload model (same as startup)
            model_size = ModelSize(server_config.get('model.size', 'medium'))
            model_path = server_config.get('model.model_path')
            
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                config = MengAIConfig(**checkpoint.get('config', {}))
                model = MengAIAdvanced(config)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model = create_meng_ai_model(model_size, vocab_size=50000)
            
            # Reload tokenizer
            tokenizer_path = server_config.get('model.tokenizer_path')
            if tokenizer_path and os.path.exists(tokenizer_path):
                tokenizer = SimpleTokenizer()
                tokenizer.load(tokenizer_path)
            else:
                tokenizer = SimpleTokenizer(vocab_size=50000, min_frequency=2)
                from simple_tokenizer import create_sample_dataset
                sample_texts = create_sample_dataset()
                tokenizer.build_vocabulary(sample_texts)
            
            # Move to device
            device = server_config.get('model.device', 'auto')
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            model.to(device)
            model.eval()
            
            logger.info("Model reloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
    
    background_tasks.add_task(reload)
    return {"message": "Model reload initiated"}

@app.get("/metrics")
async def get_metrics():
    """Get server metrics"""
    # This would typically connect to a metrics system like Prometheus
    return {
        "requests_total": len(rate_limiter.requests),
        "uptime_seconds": time.time() - server_start_time,
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent(),
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": model is not None
    }

# Utility functions
def create_meng_ai_model(size: ModelSize, vocab_size: int = 50000):
    """Create Meng AI model (imported from meng_ai_advanced)"""
    from meng_ai_advanced import create_meng_ai_model
    return create_meng_ai_model(size, vocab_size)

def run_server(config_path: Optional[str] = None):
    """Run the server"""
    global server_config
    server_config = ServerConfig(config_path)
    
    # Configure uvicorn
    uvicorn_config = uvicorn.Config(
        app,
        host=server_config.get('server.host', '0.0.0.0'),
        port=server_config.get('server.port', 8000),
        workers=server_config.get('server.workers', 1),
        log_level=server_config.get('logging.level', 'info').lower(),
        access_log=True
    )
    
    server = uvicorn.Server(uvicorn_config)
    server.run()

def create_config_file(config_path: str = "meng_ai_config.yaml"):
    """Create a sample configuration file"""
    config = {
        'model': {
            'size': 'medium',  # small, medium, large, xl
            'model_path': None,  # Path to model checkpoint
            'tokenizer_path': None,  # Path to tokenizer
            'device': 'auto'  # auto, cpu, cuda
        },
        'server': {
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 1,
            'max_request_size': 10485760,
            'timeout': 30
        },
        'security': {
            'api_key': None,  # Set to None for no authentication
            'cors_origins': ['*'],
            'trusted_hosts': ['*']
        },
        'rate_limiting': {
            'max_requests': 1000,
            'window_seconds': 3600
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration file created: {config_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Meng AI Server")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--create-config", action="store_true", help="Create sample configuration file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_config_file()
        return
    
    # Override config with command line arguments
    if args.host:
        server_config.config['server']['host'] = args.host
    if args.port:
        server_config.config['server']['port'] = args.port
    if args.workers:
        server_config.config['server']['workers'] = args.workers
    
    print("🚀 Starting Meng AI Server")
    print("=" * 50)
    print(f"Host: {server_config.get('server.host')}")
    print(f"Port: {server_config.get('server.port')}")
    print(f"Workers: {server_config.get('server.workers')}")
    print(f"Model Size: {server_config.get('model.size')}")
    print(f"Device: {server_config.get('model.device')}")
    print("=" * 50)
    
    run_server(args.config)

if __name__ == "__main__":
    main()
