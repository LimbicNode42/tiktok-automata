"""
Modern Llama-based summarizer optimized for TikTok content generation.
Simplified to use only Llama 3.2-3B-Instruct for optimal GTX 1060 6GB performance.
"""

import asyncio
import torch
import json
import re
import random
from pathlib import Path
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
    max_tokens: int = 3000  # Max tokens in summary (increased further for full 120s TTS content)
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
          # Load dynamic hooks configuration
        self.hooks_config = self._load_hooks_config()
        
        # Load voice profiles for voice selection
        self.voice_profiles = self._load_voice_profiles()
        
        logger.info(f"Initializing {self.model_name} for TikTok summarization")
        logger.info(f"Expected VRAM usage: {self.expected_vram} on {self.device}")

    def _load_hooks_config(self) -> Dict:
        """Load dynamic hooks configuration from JSON file."""
        try:
            hooks_file = Path(__file__).parent / "data" / "tiktok_hooks.json"
            with open(hooks_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load hooks config: {e}, using fallback")
            return self._get_fallback_hooks()

    def _get_fallback_hooks(self) -> Dict:
        """Fallback hooks if JSON file can't be loaded."""
        return {
            "category_hooks": {
                "ai": {
                    "hooks": ["AI just did something INSANE:"],
                    "style": "mind-blowing, tech-savvy",
                    "ctas": ["Follow for more AI chaos! What's your wildest AI prediction? 🤖"]
                },
                "tech": {
                    "hooks": ["This tech news is EVERYWHERE:"],
                    "style": "trendy, shareable", 
                    "ctas": ["Follow for daily tech! Tag someone who needs to see this! 📱"]
                }
            },
            "engagement_hooks": ["But here's where it gets crazy:"],
            "tiktok_specific_ctas": ["Follow for mind-bending tech content! 🔥"]        }
    
    def _load_voice_profiles(self) -> Dict:
        """Load Kokoro voice profiles from JSON file."""
        try:
            voice_profiles_file = Path(__file__).parent.parent / "utils" / "kokoro_voice_profiles.json"
            with open(voice_profiles_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('kokoro_voice_profiles', {})
        except Exception as e:
            logger.warning(f"Failed to load voice profiles: {e}")
            return {}

    def select_voice_for_content(self, article: Article, analysis: Dict = None) -> Dict[str, str]:
        """
        Select the most appropriate voice for the given article content.
        
        Args:
            article: The Article object containing content and category information
            analysis: Optional pre-computed content analysis
            
        Returns:
            Dict containing recommended voice information
        """
        if not self.voice_profiles:
            logger.warning("Voice profiles not loaded, using default voice")
            return {
                'voice_id': 'af_heart',
                'voice_name': 'Heart',
                'reasoning': 'Default voice (voice profiles not available)'
            }
        
        # Use existing analysis or create new one
        if analysis is None:
            analysis = self._analyze_article_content(article)
        
        category = article.category.lower()
        content_type = analysis.get('content_type', 'standard')
        
        # Try to get content type recommendations first
        try:
            voice_profiles_file = Path(__file__).parent.parent / "utils" / "kokoro_voice_profiles.json"
            with open(voice_profiles_file, 'r', encoding='utf-8') as f:
                voice_profiles_data = json.load(f)
            content_recommendations = voice_profiles_data.get('content_type_recommendations', {})
            
            if category in content_recommendations:
                rec = content_recommendations[category]
                primary_voice = rec['primary']
                reasoning = rec['reasoning']
                
                # Validate the voice exists in our profiles
                if primary_voice in self.voice_profiles:
                    return {
                        'voice_id': primary_voice,
                        'voice_name': self.voice_profiles[primary_voice]['name'],
                        'reasoning': reasoning,
                        'category_match': category,
                        'voice_profile': self.voice_profiles[primary_voice]
                    }
                
                # Try secondary voices if primary is not available
                for secondary_voice in rec.get('secondary', []):
                    if secondary_voice in self.voice_profiles:
                        return {
                            'voice_id': secondary_voice,
                            'voice_name': self.voice_profiles[secondary_voice]['name'],
                            'reasoning': f"{reasoning} (using secondary choice)",
                            'category_match': category,
                            'voice_profile': self.voice_profiles[secondary_voice]
                        }
        except Exception as e:
            logger.debug(f"Error loading content recommendations: {e}")
        
        # Fallback: select based on content analysis and quality
        return self._select_voice_fallback(article, analysis)

    def _select_voice_fallback(self, article: Article, analysis: Dict) -> Dict[str, str]:
        """
        Fallback voice selection based on content analysis and voice quality.
        """
        # Get high-quality voices
        high_quality_voices = ['af_heart', 'af_bella', 'bf_emma', 'am_michael', 'am_fenrir', 'bm_george']
        
        # Content-based selection logic
        content_lower = article.content.lower()
        title_lower = article.title.lower()
        
        # Tech content
        if any(word in content_lower + title_lower for word in ['ai', 'tech', 'software', 'algorithm', 'digital']):
            if 'af_nicole' in self.voice_profiles:
                return {
                    'voice_id': 'af_nicole',
                    'voice_name': self.voice_profiles['af_nicole']['name'],
                    'reasoning': 'Tech-focused content detected',
                    'fallback_selection': True
                }
        
        # High-energy content
        if analysis.get('is_breakthrough') or any(word in content_lower for word in ['breaking', 'amazing', 'incredible']):
            if 'af_bella' in self.voice_profiles:
                return {
                    'voice_id': 'af_bella',
                    'voice_name': self.voice_profiles['af_bella']['name'],
                    'reasoning': 'High-energy content detected',
                    'fallback_selection': True
                }
        
        # Business/professional content
        if any(word in content_lower for word in ['business', 'company', 'investment', 'funding']):
            if 'am_michael' in self.voice_profiles:
                return {
                    'voice_id': 'am_michael',
                    'voice_name': self.voice_profiles['am_michael']['name'],
                    'reasoning': 'Professional/business content detected',
                    'fallback_selection': True
                }
        
        # Default to a high-quality versatile voice
        for voice_id in ['af_heart', 'af_kore', 'am_michael']:
            if voice_id in self.voice_profiles:
                return {
                    'voice_id': voice_id,
                    'voice_name': self.voice_profiles[voice_id]['name'],
                    'reasoning': 'Default versatile voice selection',
                    'fallback_selection': True
                }
        
        # Final fallback
        return {
            'voice_id': 'af_heart',
            'voice_name': 'Heart',
            'reasoning': 'Final fallback voice'
        }

    def get_available_voices_for_category(self, category: str) -> List[Dict]:
        """
        Get a list of recommended voices for a specific content category.
        
        Args:
            category: Content category (e.g., 'ai', 'tech', 'business')
            
        Returns:
            List of voice information dictionaries
        """
        if not self.voice_profiles:
            return []
        
        try:
            # Try to load content type recommendations
            voice_profiles_file = Path(__file__).parent.parent / "utils" / "kokoro_voice_profiles.json"
            with open(voice_profiles_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                content_recommendations = data.get('content_type_recommendations', {})
            
            if category in content_recommendations:
                rec = content_recommendations[category]
                voices = []
                
                # Add primary voice
                primary = rec['primary']
                if primary in self.voice_profiles:
                    voices.append({
                        'voice_id': primary,
                        'voice_name': self.voice_profiles[primary]['name'],
                        'description': self.voice_profiles[primary]['description'],
                        'personality': self.voice_profiles[primary]['personality'],
                        'priority': 'primary',
                        'reasoning': rec['reasoning']
                    })
                
                # Add secondary voices
                for secondary in rec.get('secondary', []):
                    if secondary in self.voice_profiles:
                        voices.append({
                            'voice_id': secondary,
                            'voice_name': self.voice_profiles[secondary]['name'],
                            'description': self.voice_profiles[secondary]['description'],
                            'personality': self.voice_profiles[secondary]['personality'],
                            'priority': 'secondary',
                            'reasoning': rec['reasoning']
                        })
                
                return voices
                
        except Exception as e:
            logger.debug(f"Error getting voices for category {category}: {e}")
        
        return []
    
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
                # 🚀 Optimized generation parameters
                result = self.pipeline(
                    test_prompt,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    num_beams=1,  # Disable beam search for speed
                    torch_dtype=torch.float16,  # Use half precision
                    use_cache=True,  # Enable KV cache
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            benchmark_time = time.time() - start_time
            tokens_per_second = 50 / benchmark_time
            
            logger.info(f"Benchmark: {tokens_per_second:.1f} tokens/second")
            
        except Exception as e:
            logger.warning(f"Benchmark failed: {str(e)}")

    def _analyze_article_content(self, article: Article) -> Dict[str, any]:
        """Analyze article content to determine appropriate hooks and style."""
        content_lower = article.content.lower()
        title_lower = article.title.lower()
        full_text = f"{title_lower} {content_lower}"
        
        analysis = {
            'content_type': 'standard',
            'urgency_level': 'medium',
            'controversy_score': 0,
            'has_funding': False,
            'funding_amount': None,
            'is_breakthrough': False,
            'is_partnership': False,
            'is_acquisition': False,
            'is_first_person': False
        }
        
        # Check for first-person indicators
        first_person_patterns = [
            r'\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b',
            r'\bi\s+(am|was|have|had|will|can|should|would|think|believe|feel)',
            r'\bmy\s+(thoughts|experience|opinion|view|perspective)',
            r'\bwe\s+(are|were|have|had|will|can|should|would|think|believe)',
            r'\bin\s+my\s+(opinion|view|experience)',
            r'\bi\s+(wrote|published|created|built|founded)'
        ]
        
        first_person_matches = 0
        for pattern in first_person_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            first_person_matches += len(matches)
        
        # If we find 3+ first-person indicators, treat as first-person content
        if first_person_matches >= 3:
            analysis['is_first_person'] = True
            analysis['content_type'] = 'first_person'
        
        # Check for funding/money mentions
        funding_patterns = [r'\$(\d+(?:\.\d+)?)\s*(?:billion|million|b|m)', r'raised\s+\$', r'funding\s+round', r'investment']
        for pattern in funding_patterns:
            match = re.search(pattern, full_text)
            if match:
                analysis['has_funding'] = True
                analysis['content_type'] = 'high_funding'
                if match.groups():
                    analysis['funding_amount'] = match.group(1)
                break
        
        # Check for controversy indicators
        controversy_words = ['controversy', 'backlash', 'criticism', 'outrage', 'scandal', 'debate', 'divided', 'polarizing']
        analysis['controversy_score'] = sum(1 for word in controversy_words if word in full_text)
        if analysis['controversy_score'] > 0:
            analysis['content_type'] = 'controversy'
        
        # Check for breakthrough/innovation
        breakthrough_words = ['breakthrough', 'revolutionary', 'game-changing', 'unprecedented', 'historic', 'first time', 'never before']
        if any(word in full_text for word in breakthrough_words):
            analysis['is_breakthrough'] = True
            analysis['content_type'] = 'breakthrough'
        
        # Check for partnerships
        partnership_words = ['partnership', 'collaboration', 'teams up', 'joins forces', 'alliance', 'together']
        if any(word in full_text for word in partnership_words):
            analysis['is_partnership'] = True
            analysis['content_type'] = 'partnership'
        
        # Check for acquisitions
        acquisition_words = ['acquires', 'acquisition', 'buys', 'purchased', 'bought', 'merger']
        if any(word in full_text for word in acquisition_words):
            analysis['is_acquisition'] = True
            analysis['content_type'] = 'acquisition'
        
        # Check for major announcements
        announcement_words = ['announces', 'reveals', 'unveils', 'launches', 'introduces', 'release']
        if any(word in title_lower for word in announcement_words):
            analysis['content_type'] = 'major_announcement'
        
        # Determine urgency level
        urgent_words = ['breaking', 'urgent', 'just in', 'developing', 'alert', 'emergency']
        if any(word in full_text for word in urgent_words):
            analysis['urgency_level'] = 'high'
        
        return analysis

    def _select_dynamic_hook(self, article: Article, analysis: Dict) -> str:
        """Select the most appropriate hook based on article analysis."""
        category = article.category
        hooks_data = self.hooks_config.get('category_hooks', {})
        content_hooks = self.hooks_config.get('content_based_hooks', {})
        
        # Get category-specific hooks as fallback
        category_data = hooks_data.get(category, hooks_data.get('tech', {}))
        category_hooks = category_data.get('hooks', ["This tech news is EVERYWHERE:"])
        
        # Try to get content-specific hook first
        content_type = analysis.get('content_type', 'standard')
        if content_type in content_hooks:
            content_specific_hooks = content_hooks[content_type]
            selected_hook = random.choice(content_specific_hooks)
            
            # Replace placeholders if needed
            if analysis.get('funding_amount') and '${amount}' in selected_hook:
                amount = analysis['funding_amount']
                # Format amount nicely
                if float(amount) >= 1000:
                    amount = f"{float(amount)/1000:.1f}B" if float(amount) >= 1000 else f"{amount}M"
                selected_hook = selected_hook.replace('${amount}', f"${amount}")
            
            return selected_hook
        
        # Fall back to category-specific hook
        return random.choice(category_hooks)

    def _select_dynamic_cta(self, article: Article) -> str:
        """Select appropriate call-to-action based on article category."""
        category = article.category
        hooks_data = self.hooks_config.get('category_hooks', {})
        tiktok_ctas = self.hooks_config.get('tiktok_specific_ctas', [])
        
        # Get category-specific CTAs
        category_data = hooks_data.get(category, hooks_data.get('tech', {}))
        category_ctas = category_data.get('ctas', [])
        
        # 70% chance to use category-specific CTA, 30% chance to use generic TikTok CTA
        if category_ctas and random.random() < 0.7:
            return random.choice(category_ctas)
        else:
            return random.choice(tiktok_ctas) if tiktok_ctas else "Follow for more amazing content! 🔥"

    def create_tiktok_prompt(self, article: Article, target_duration: int = None) -> str:
        """Create engaging TikTok-optimized prompts using dynamic hooks."""
        duration = target_duration or self.config.target_duration
        
        # Analyze article content for appropriate hooks
        analysis = self._analyze_article_content(article)
        
        # Select dynamic hook and CTA
        selected_hook = self._select_dynamic_hook(article, analysis)
        selected_cta = self._select_dynamic_cta(article)
        selected_engagement_hook = random.choice(self.hooks_config.get('engagement_hooks', ["But here's where it gets crazy:"]))
          # Get category style information
        category_data = self.hooks_config.get('category_hooks', {}).get(
            article.category, 
            self.hooks_config.get('category_hooks', {}).get('tech', {})
        )
        style_description = category_data.get('style', 'engaging, energetic')
          # Determine content approach based on analysis
        if analysis.get('is_first_person', False):
            content_approach = """
IMPORTANT: This article is written in first person. DO NOT copy the first-person perspective. Instead:
- Analyze what the author is saying from a third-person perspective
- Use phrases like "The author claims...", "According to the writer...", "The CEO explains..."
- Maintain analytical distance while keeping the TikTok energy
- Focus on the implications and significance of their statements"""
        else:
            content_approach = """
- Present the information with TikTok energy and enthusiasm
- Use engaging storytelling techniques"""        # Use modern chat template format for Llama 3.2
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a viral TikTok content creator specializing in tech news. Create engaging {duration}-second TikTok scripts that hook viewers immediately and keep them watching until the end.

Your style should be: {style_description}
Target audience: Tech-curious Gen Z and Millennials

CRITICAL: Output ONLY the TikTok script content. Do NOT include any introductory phrases like "Here's the script:", "Here is a script:", or similar. Start directly with the actual content.<|eot_id|><|start_header_id|>user<|end_header_id|>

Create a {duration}-second TikTok script from this article:

Title: {article.title}
Category: {article.category}
Content: {article.content[:1000]}{"..." if len(article.content) > 1000 else ""}

Requirements:
- Start with: "{selected_hook}"
- Target EXACTLY {duration} seconds when read aloud (approximately 500-600 words for full 2-minute content)
- Hook viewers in first 3 seconds{content_approach}
- Use short, punchy sentences with dramatic pauses
- Include the engagement hook "{selected_engagement_hook}" somewhere in the middle
- Include 3-4 surprising facts or "wait, what?" moments throughout
- Build tension/curiosity and maintain high energy
- Add multiple engagement hooks and dramatic moments
- End with: "{selected_cta}"
- Keep it conversational and energetic
- Use strategic pauses for emphasis and dramatic effect
- Make it feel like a 2-minute story that flies by
- DO NOT include timestamps or time markers
- DO NOT use emojis in the main script
- DO NOT include any introductory text like "Here's the script:" or "Here is a script based on..."

Output ONLY the script content that would be spoken in the TikTok video. Start immediately with "{selected_hook}" and end with "{selected_cta}".<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    async def summarize_for_tiktok(self, article: Article, target_duration: int = None, include_voice_recommendation: bool = False) -> Optional[str]:
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
            
            # Optionally include voice recommendation
            if include_voice_recommendation:
                analysis = self._analyze_article_content(article)
                voice_recommendation = self.select_voice_for_content(article, analysis)
                return {
                    'summary': summary,
                    'voice_recommendation': voice_recommendation,                'content_analysis': analysis            }
            
            return summary
            
        except Exception as e:
            logger.error(f"TikTok summarization failed: {str(e)}")
            return None

    async def generate_tiktok_summary(self, content: str, title: str, url: str, target_duration: int = None) -> Optional[str]:
        """Generate TikTok summary from individual content components."""
        # Create an Article object from the provided arguments
        from scraper.newsletter_scraper import Article
        
        article = Article(
            title=title,
            content=content,
            url=url,
            published_date="",  # Not critical for summarization
            category="general"  # Default category
        )
        
        # Use the existing summarize_for_tiktok method
        return await self.summarize_for_tiktok(article, target_duration)

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
        
        # Remove leading quotes that might wrap the entire response
        summary = re.sub(r'^[\"\']', '', summary)
        
        # Remove introductory phrases that appear at the beginning
        intro_patterns = [
            r'^Here is the script:\s*',
            r'^Here\'s the script:\s*',
            r'^Here is a \d+-second TikTok script based on the provided article:\s*',
            r'^Here is a script for a \d+-second TikTok video based on the provided article:\s*',
            r'^Here is a script based on the provided article:\s*',
            r'^Here is a script that meets the requirements:\s*',
            r'^Here is a script:\s*',
            r'^Here\'s a script:\s*',
            r'^Script:\s*',
            r'^TikTok script:\s*',
            r'^Here\'s your TikTok script:\s*',
            r'^Here is your TikTok script:\s*',
        ]
        
        for pattern in intro_patterns:
            summary = re.sub(pattern, '', summary, flags=re.IGNORECASE)
        
        # Clean up any remaining leading quotes or colons
        summary = re.sub(r'^[\"\':]*\s*', '', summary)
        
        # Remove timestamps - comprehensive patterns (including at the beginning)
        timestamp_patterns = [
            r'^\[\d+s\]\s*',            # [0s] at the beginning
            r'^\[[\d\s\-:]+\]\s*',      # [10s-15s] at the beginning
            r'^\([\d\s\-:]+\)\s*',      # (20s-25s) at the beginning
            r'\*\*\[[\d\s\-:]+\]\*\*',  # **[0s-3s]** anywhere
            r'\[[\d\s\-:]+\]',          # [10s-15s] anywhere
            r'\([\d\s\-:]+\)',          # (20s-25s) anywhere
            r'\d+s\s*-\s*\d+s',         # 30s - 35s
            r'\d+:\d+\s*-\s*\d+:\d+',   # 1:20 - 1:25
            r'at\s+\d+\s*seconds?',     # at 45 seconds
            r'from\s+\d+\s*to\s+\d+\s*seconds?',  # from 10 to 15 seconds
            r'\d+\s*sec\s*-\s*\d+\s*sec',  # 5 sec - 10 sec
            r'timestamp:\s*\d+',        # timestamp: 30
            r'\[\d+\]',                 # [1], [2], etc. (numbered markers)
        ]
        
        for pattern in timestamp_patterns:
            summary = re.sub(pattern, '', summary, flags=re.IGNORECASE)
        
        # Remove emojis
        summary = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000027BF]+', '', summary)
        
        # Clean up extra whitespace
        summary = " ".join(summary.split())
        
        # Ensure it doesn't exceed reasonable length (increased for 120s videos)
        if len(summary) > 1800:  # Increased from 1200 to 1800 for 120s target
            sentences = summary.split(". ")
            # Keep sentences until we hit a reasonable length
            truncated = []
            char_count = 0
            for sentence in sentences:
                if char_count + len(sentence) < 1600:  # Increased from 1100 to 1600
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
            
            # Analyze content for voice selection
            analysis = self._analyze_article_content(article)
            
            # Generate summary
            summary = await self.summarize_for_tiktok(article)
            
            # Select appropriate voice
            voice_recommendation = self.select_voice_for_content(article, analysis)
            
            result = {
                'article': article,
                'tiktok_summary': summary,
                'voice_recommendation': voice_recommendation,
                'content_analysis': analysis,
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
