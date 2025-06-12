"""
Modern Llama-based summarizer optimized for TikTok content generation.
Simplified to use only Llama 3.2-3B-Instruct for optimal GTX 1060 6GB performance.
"""

import asyncio
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline
)
from typing import Optional, Dict, List
from loguru import logger
from dataclasses import dataclass
import time

try:
    from ..scraper.newsletter_scraper import Article
except ImportError:
    # Fallback for when running as standalone script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.scraper.newsletter_scraper import Article


@dataclass
class TikTokSummaryConfig:
    """Configuration for TikTok summary generation."""
    target_duration: int = 120  # Target seconds for TikTok video
    max_tokens: int = 300  # Max tokens in summary (increased for longer content)
    temperature: float = 0.7  # Creativity level
    top_p: float = 0.9  # Nucleus sampling
    use_gpu: bool = True  # Use GPU acceleration


class LlamaSummarizer:
    """
    Llama 3.2-3B-Instruct summarizer optimized for RTX 3070 and TikTok content.
    Fixed to the optimal model for your hardware - no complex model selection needed.
    """
    
    def __init__(self, config: TikTokSummaryConfig = None):
        """
        Initialize the Llama summarizer with the 3.2-3B instruction-tuned model.
          Args:
            config: TikTok summary configuration
        """
        self.config = config or TikTokSummaryConfig()
        # Use Llama 3.2-3B-Instruct - optimal for RTX 3070 8GB
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Meta's efficient 3B instruction-tuned model
        self.description = "Llama 3.2-3B Instruct model optimized for RTX 3070 8GB"
        self.expected_vram = "~1.8GB"
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        logger.info(f"Initializing {self.model_name} for TikTok summarization")
        logger.info(f"Expected VRAM usage: {self.expected_vram} on {self.device}")
    
    def _setup_authentication(self):
        """Setup Hugging Face authentication for accessing Llama models."""
        import os
        from huggingface_hub import login
        
        # Try multiple authentication methods
        token = None
        
        # Method 1: Environment variable
        token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
        
        if token:
            try:
                login(token=token)
                logger.info("✅ Authenticated with Hugging Face using environment variable")
                return
            except Exception as e:
                logger.warning(f"Failed to authenticate with environment token: {e}")
        
        # Method 2: Check if already logged in
        try:
            from huggingface_hub import whoami
            user_info = whoami()
            logger.info(f"✅ Already authenticated as: {user_info['name']}")
            return
        except Exception:
            pass
        
        # Method 3: Look for .env file
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('HF_TOKEN='):
                        token = line.split('=', 1)[1].strip().strip('"')
                        try:
                            login(token=token)
                            logger.info("✅ Authenticated with Hugging Face using .env file")
                            return
                        except Exception as e:
                            logger.warning(f"Failed to authenticate with .env token: {e}")
        
        # If we get here, authentication failed
        logger.error("❌ Hugging Face authentication failed!")
        logger.error("Please set up authentication using one of these methods:")
        logger.error("1. Set HF_TOKEN environment variable")
        logger.error("2. Run: huggingface-cli login")
        logger.error("3. Create .env file with HF_TOKEN=your_token")
        logger.error("4. Run: python setup_hf_auth.py")
        raise ValueError("Hugging Face authentication required for Llama models")
    
    async def initialize(self):
        """Load Llama model with hardware-optimized settings and authentication."""
        start_time = time.time()
        logger.info(f"Loading {self.model_name} on {self.device}...")
        
        try:
            # Setup authentication
            self._setup_authentication()
            
            # Load tokenizer (Llama uses standard tokenizer)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations for RTX 3070 8GB
            # Llama 3.2-3B uses float16 for optimal performance
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None
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
        """Create engaging TikTok-optimized prompts for Llama 3.2."""
        duration = target_duration or self.config.target_duration
        
        # Category-specific hooks and styles
        category_styles = {
            'ai': {
                'hook': "AI just did something INSANE:",
                'style': "mind-blowing, tech-savvy",
                'cta': "What AI breakthrough shocked you most? Drop it below!"
            },
            'big_tech': {
                'hook': "Big Tech just changed EVERYTHING:",
                'style': "dramatic, insider knowledge",
                'cta': "Are you Team Apple or Team Google? Let me know!"
            },
            'dev': {
                'hook': "Developers, this will blow your mind:",
                'style': "technical but accessible, excited",
                'cta': "Which programming tip changed your life? Share it!"
            },
            'science': {
                'hook': "Scientists just discovered something WILD:",
                'style': "fascinating, educational",
                'cta': "What science fact still amazes you? Tell me!"
            },
            'crypto': {
                'hook': "Crypto world is going CRAZY:",
                'style': "hype, financial excitement",
                'cta': "Diamond hands or paper hands? Comment below!"
            },
            'tech': {
                'hook': "This tech news is EVERYWHERE:",
                'style': "trendy, shareable",
                'cta': "Tag someone who needs to see this!"
            }
        }
        
        style_info = category_styles.get(article.category, category_styles['tech'])
        
        # Use modern chat template format for Llama 3.2
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
- DO NOT include timestamps or time markers
- DO NOT use emojis

Format as a natural speech script with clear paragraph breaks for emphasis.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    async def summarize_for_tiktok(self, article: Article, target_duration: int = None) -> Optional[str]:
        """Generate TikTok-optimized summary using Llama 3.2-3B."""
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
        import re
        
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
        
        # Remove timestamps (e.g., **[0s - 3s]**, [10s-15s], etc.)
        summary = re.sub(r'\*\*\[[\d\s\-:]+\]\*\*', '', summary)
        summary = re.sub(r'\[[\d\s\-:]+\]', '', summary)
        
        # Remove emojis
        summary = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000027BF]+', '', summary)
        
        # Clean up extra whitespace
        summary = " ".join(summary.split())
        
        # Ensure it doesn't exceed reasonable length (increased for 120s videos)
        if len(summary) > 800:
            sentences = summary.split(". ")
            # Keep sentences until we hit a reasonable length
            truncated = []
            char_count = 0
            for sentence in sentences:
                if char_count + len(sentence) < 750:
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
            'model_name': self.model_name,
            'device': self.device,
            'vram_usage': self.expected_vram,
            'description': self.description,
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


# Simplified factory function
def create_tiktok_summarizer() -> LlamaSummarizer:
    """
    Create a TikTok summarizer optimized for GTX 1060 6GB.
    
    Returns:
        LlamaSummarizer configured with Llama 3.2-3B-Instruct
    """
    return LlamaSummarizer()
