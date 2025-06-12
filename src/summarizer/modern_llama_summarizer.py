"""
Modern Llama-based summarizer optimized for TikTok content generation.
Supports Llama 3.3, 3.2, and 4.x models with hardware-optimized quantization.
"""

import asyncio
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from typing import Optional, Dict, List
from loguru import logger
from dataclasses import dataclass
import json
import time
from pathlib import Path

from ..scraper.newsletter_scraper import Article


@dataclass
class TikTokSummaryConfig:
    """Configuration for TikTok summary generation."""
    target_duration: int = 60  # Target seconds for TikTok video
    max_tokens: int = 150  # Max tokens in summary
    temperature: float = 0.7  # Creativity level
    top_p: float = 0.9  # Nucleus sampling
    use_gpu: bool = True  # Use GPU acceleration


class ModernLlamaSummarizer:
    """
    Modern Llama-based summarizer optimized for GTX 1060 6GB and TikTok content.
    Uses Llama 3.2-3B-Instruct for the best speed/quality balance.
    """
    
    def __init__(self, config: TikTokSummaryConfig = None):
        """
        Initialize the Llama summarizer with the balanced 3B model.
        
        Args:
            config: TikTok summary configuration
        """
        self.config = config or TikTokSummaryConfig()
        
        # Fixed to the optimal model for your hardware
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.description = "Balanced speed/quality model optimized for GTX 1060 6GB"
        self.expected_vram = "~1.8GB"
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        
        logger.info(f"Initializing Llama 3.2-3B-Instruct for TikTok summarization")
        logger.info(f"Expected VRAM usage: {self.expected_vram} on {self.device}")
    
    async def initialize(self):
        """Load model with hardware-optimized settings."""
        start_time = time.time()
        logger.info(f"Loading {self.model_name} on {self.device}...")
          try:
            # No quantization needed for 3B model on 6GB GPU - runs efficiently
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations for GTX 1060 6GB
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": {"": 0} if self.device == "cuda" else None
            }
              self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            
            load_time = time.time() - start_time
            logger.success(f"Model loaded successfully in {load_time:.1f}s")
            
            # Test generation speed
            await self._benchmark_speed()
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    async def _benchmark_speed(self):
        """Quick speed benchmark."""
        try:
            test_prompt = "Summarize this: AI breakthrough in computing."
            start_time = time.time()
            
            with torch.no_grad():
                result = self.pipeline(
                    test_prompt,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            benchmark_time = time.time() - start_time
            tokens_per_second = 50 / benchmark_time
            
            logger.info(f"Benchmark: {tokens_per_second:.1f} tokens/second")
            
        except Exception as e:
            logger.warning(f"Benchmark failed: {str(e)}")
    
    def create_tiktok_prompt(self, article: Article, target_duration: int = None) -> str:
        """Create engaging TikTok-optimized prompts for latest Llama models."""
        duration = target_duration or self.config.target_duration
        
        # Category-specific hooks and styles
        category_styles = {
            'ai': {
                'hook': "🤖 AI just did something INSANE:",
                'style': "mind-blowing, tech-savvy",
                'cta': "What AI breakthrough shocked you most? Drop it below! 👇"
            },
            'big_tech': {
                'hook': "🚨 Big Tech just changed EVERYTHING:",
                'style': "dramatic, insider knowledge",
                'cta': "Are you Team Apple or Team Google? Let me know! 💬"
            },
            'dev': {
                'hook': "💻 Developers, this will blow your mind:",
                'style': "technical but accessible, excited",
                'cta': "Which programming tip changed your life? Share it! 🔥"
            },
            'science': {
                'hook': "🧬 Scientists just discovered something WILD:",
                'style': "fascinating, educational",
                'cta': "What science fact still amazes you? Tell me! 🤯"
            },
            'crypto': {
                'hook': "💰 Crypto world is going CRAZY:",
                'style': "hype, financial excitement",
                'cta': "Diamond hands or paper hands? Comment below! 💎"
            },
            'tech': {
                'hook': "🔥 This tech news is EVERYWHERE:",
                'style': "trendy, shareable",
                'cta': "Tag someone who needs to see this! 👆"
            }
        }
        
        style_info = category_styles.get(article.category, category_styles['tech'])
        
        # Use modern chat template format for Llama 3.3/4.x
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a viral TikTok content creator specializing in tech news. Create engaging {duration}-second TikTok scripts that hook viewers immediately and keep them watching until the end.

Your style should be: {style_info['style']}
Target audience: Tech-curious Gen Z and Millennials<|eot_id|><|start_header_id|>user<|end_header_id|>

Create a {duration}-second TikTok script from this article:

Title: {article.title}
Category: {article.category}
Content: {article.content[:1000]}{"..." if len(article.content) > 1000 else ""}

Requirements:
- Start with: "{style_info['hook']}"
- Hook viewers in first 3 seconds
- Use short, punchy sentences
- Include 1-2 surprising facts or "wait, what?" moments
- Build tension/curiosity throughout
- End with: "{style_info['cta']}"
- Keep it conversational and energetic
- Target exactly {duration} seconds when read aloud
- Use strategic pauses for emphasis

Format as a natural speech script with [PAUSE] markers where needed.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    async def summarize_for_tiktok(self, article: Article, target_duration: int = None) -> Optional[str]:
        """Generate TikTok-optimized summary using modern Llama models."""
        if not self.pipeline:
            await self.initialize()
        
        duration = target_duration or self.config.target_duration
        
        try:
            start_time = time.time()
            
            # Create optimized prompt
            prompt = self.create_tiktok_prompt(article, duration)
            
            # Generate with optimized parameters
            with torch.no_grad():
                result = self.pipeline(
                    prompt,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract and clean the response
            generated_text = result[0]['generated_text']
            
            # Extract just the assistant's response
            if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
                summary = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            else:
                # Fallback for other models
                summary = generated_text[len(prompt):] if len(generated_text) > len(prompt) else generated_text
            
            # Clean up the summary
            summary = self._clean_summary(summary)
            
            generation_time = time.time() - start_time
            logger.info(f"Generated TikTok summary in {generation_time:.2f}s")
            
            return summary
            
        except Exception as e:
            logger.error(f"TikTok summarization failed: {str(e)}")
            return None
    
    def _clean_summary(self, summary: str) -> str:
        """Clean and format the generated summary."""
        # Remove common artifacts
        summary = summary.strip()
        
        # Remove end tokens and artifacts
        cleanup_patterns = [
            "<|eot_id|>",
            "<|end_of_text|>",
            "[INST]",
            "[/INST]",
        ]
        
        for pattern in cleanup_patterns:
            summary = summary.replace(pattern, "")
        
        # Clean up extra whitespace
        summary = " ".join(summary.split())
        
        # Ensure it doesn't exceed reasonable length
        if len(summary) > 500:
            sentences = summary.split(". ")
            # Keep sentences until we hit a reasonable length
            truncated = []
            char_count = 0
            for sentence in sentences:
                if char_count + len(sentence) < 450:
                    truncated.append(sentence)
                    char_count += len(sentence)
                else:
                    break
            summary = ". ".join(truncated)
            if not summary.endswith('.'):
                summary += "."
        
        return summary.strip()
    
    async def batch_summarize(self, articles: List[Article]) -> List[Dict]:
        """Efficiently process multiple articles."""
        if not self.pipeline:
            await self.initialize()
        
        results = []
        total_start = time.time()
        
        logger.info(f"Starting batch summarization of {len(articles)} articles")
        
        for i, article in enumerate(articles, 1):
            if article.content_extraction_status != "success":
                logger.warning(f"Skipping article {i} '{article.title}' - extraction status: {article.content_extraction_status}")
                continue
            
            logger.info(f"Processing article {i}/{len(articles)}: {article.title[:50]}...")
            
            summary = await self.summarize_for_tiktok(article)
            
            result = {
                'article': article,
                'tiktok_summary': summary,
                'success': summary is not None,
                'processing_order': i
            }
            
            results.append(result)
            
            # Brief pause to prevent overheating
            if i % 5 == 0:
                await asyncio.sleep(1)
        
        total_time = time.time() - total_start
        successful = sum(1 for r in results if r['success'])
        
        logger.success(f"Batch complete: {successful}/{len(results)} successful in {total_time:.1f}s")
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the current model configuration."""
        return {
            'tier': self.model_tier,
            'model_name': self.model_name,
            'device': self.device,
            'vram_usage': self.model_info['vram_usage'],
            'speed': self.model_info['speed'],
            'description': self.model_info['description'],
            'loaded': self.pipeline is not None
        }
    
    async def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            del self.pipeline
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model resources cleaned up")


# Factory function for easy model selection
def create_summarizer_for_hardware(hardware_profile: str = "gtx_1060_6gb") -> ModernLlamaSummarizer:
    """
    Create an optimized summarizer based on hardware profile.
    
    Args:
        hardware_profile: "gtx_1060_6gb", "low_end", "high_end"
    """
    if hardware_profile == "gtx_1060_6gb":
        # Perfect for your setup!
        return ModernLlamaSummarizer("balanced")  # Llama 3.2-3B
    elif hardware_profile == "low_end":
        return ModernLlamaSummarizer("ultra_fast")  # Llama 3.2-1B  
    elif hardware_profile == "high_end":
        return ModernLlamaSummarizer("high_quality")  # Llama 3.3-70B
    else:
        return ModernLlamaSummarizer("balanced")
