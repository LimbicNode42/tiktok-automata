"""
Kokoro TTS Engine for TikTok Automata.
High-quality, fast text-to-speech using the lightweight Kokoro model.
"""

import asyncio
import os
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
import soundfile as sf
import torch
from loguru import logger

try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    logger.warning("Kokoro not installed. Install with: pip install kokoro>=0.9.4 soundfile")
except AttributeError as e:
    KOKORO_AVAILABLE = False
    logger.warning(f"Kokoro installation issue: {e}. This may be related to eSpeak NG dependencies.")
except Exception as e:
    KOKORO_AVAILABLE = False
    logger.warning(f"Kokoro initialization error: {e}")

try:
    from ..scraper.newsletter_scraper import Article
except ImportError:
    # Fallback for standalone usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.scraper.newsletter_scraper import Article


@dataclass
class TTSConfig:
    """Configuration for TTS generation."""
    voice: str = 'af_heart'  # Default voice
    speed: float = 1.0  # Speech speed (0.5-2.0)
    language: str = 'a'  # 'a' for American English
    sample_rate: int = 24000  # Kokoro's native sample rate
    output_format: str = 'wav'  # Output audio format
    split_pattern: str = r'\n+'  # Pattern for splitting long text
    max_length: int = 1000  # Max chars per segment to avoid memory issues
    
    # Quality settings
    use_gpu: bool = True
    batch_size: int = 1
    
    # TikTok-specific settings
    target_duration: float = 60.0  # Target duration in seconds
    normalize_audio: bool = True
    add_silence_padding: bool = True
    padding_seconds: float = 0.5


class KokoroTTSEngine:
    """
    Kokoro TTS Engine optimized for TikTok content generation.
    
    Features:
    - High-quality 82M parameter model
    - Multiple voice options
    - Fast generation (real-time or faster)
    - Multiple language support
    - GPU acceleration
    """
    
    def __init__(self, config: TTSConfig = None):
        self.config = config or TTSConfig()
        self.pipeline: Optional[KPipeline] = None
        self.initialized = False
        
        # Voice options available in Kokoro
        self.available_voices = {
            'af_heart': 'Warm, emotional female voice',
            'af_sky': 'Clear, professional female voice', 
            'af_sunset': 'Soft, gentle female voice',
            'am_adam': 'Strong, confident male voice',
            'am_michael': 'Professional, news-anchor male voice',
            'bf_emma': 'British female voice',
            'bf_isabella': 'Elegant British female voice',
            'bm_george': 'Distinguished British male voice',
            'bm_lewis': 'Casual British male voice'
        }
        
        # Language codes
        self.language_codes = {
            'american_english': 'a',
            'british_english': 'b', 
            'spanish': 'e',
            'french': 'f',
            'hindi': 'h',
            'italian': 'i',
            'japanese': 'j',
            'portuguese': 'p',
            'chinese': 'z'
        }
        
        logger.info(f"Initialized Kokoro TTS Engine")
        logger.info(f"Available voices: {len(self.available_voices)}")
        logger.info(f"Default voice: {self.config.voice} ({self.available_voices.get(self.config.voice, 'Unknown')})")
    
    async def initialize(self):
        """Initialize the Kokoro TTS pipeline."""
        if not KOKORO_AVAILABLE:
            raise ImportError("Kokoro TTS not available. Install with: pip install kokoro>=0.9.4 soundfile")
        
        if self.initialized:
            logger.info("Kokoro TTS already initialized")
            return
        
        start_time = time.time()
        logger.info(f"Initializing Kokoro TTS with language '{self.config.language}'...")
        
        try:
            # Initialize the Kokoro pipeline
            self.pipeline = KPipeline(lang_code=self.config.language)
            
            # Test generation to ensure everything works
            await self._test_generation()
            
            self.initialized = True
            
            init_time = time.time() - start_time
            logger.success(f"Kokoro TTS initialized successfully in {init_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}")
            raise
    
    async def _test_generation(self):
        """Test generation with a short phrase."""
        try:
            test_text = "Testing Kokoro TTS initialization."
            generator = self.pipeline(
                test_text, 
                voice=self.config.voice,
                speed=self.config.speed,
                split_pattern=self.config.split_pattern
            )
            
            # Generate one sample to test
            for i, (gs, ps, audio) in enumerate(generator):
                if i == 0:  # Just test the first segment
                    logger.debug(f"Test generation successful: {len(audio)} samples")
                    break
                    
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            raise
    
    def list_voices(self) -> Dict[str, str]:
        """Get available voices and their descriptions."""
        return self.available_voices.copy()
    
    def set_voice(self, voice: str):
        """Change the current voice."""
        if voice in self.available_voices:
            self.config.voice = voice
            logger.info(f"Voice changed to: {voice} ({self.available_voices[voice]})")
        else:
            logger.warning(f"Voice '{voice}' not available. Available: {list(self.available_voices.keys())}")
    
    def _prepare_text(self, text: str) -> str:
        """Prepare text for TTS generation."""
        # Remove or replace problematic characters
        text = text.replace('"', "'")  # Replace quotes
        text = text.replace('`', "'")  # Replace backticks
        text = text.replace('\t', ' ')  # Replace tabs
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        # Remove markdown-style formatting
        text = text.replace('**', '')  # Bold
        text = text.replace('*', '')   # Italic
        text = text.replace('_', ' ')  # Underscores
        text = text.replace('#', 'hashtag ')  # Hashtags
        
        # Handle time markers for TikTok scripts
        import re
        text = re.sub(r'\[(\d+)s?[-–](\d+)s?\]', r'', text)  # Remove [0s-3s] markers
        text = re.sub(r'\*\*\[.*?\]\*\*', '', text)  # Remove **[0s-3s]** markers
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _split_text_intelligently(self, text: str) -> List[str]:
        """Split text into manageable chunks for TTS."""
        if len(text) <= self.config.max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed max length, start a new chunk
            if len(current_chunk) + len(sentence) + 2 > self.config.max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + "."
                else:
                    # Sentence is too long by itself, split it
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 > self.config.max_length:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = word
                            else:
                                # Single word is too long, just add it
                                chunks.append(word)
                        else:
                            temp_chunk += " " + word if temp_chunk else word
                    
                    if temp_chunk:
                        current_chunk = temp_chunk + "."
            else:
                current_chunk += " " + sentence + "." if current_chunk else sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def generate_audio(
        self, 
        text: str, 
        output_path: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None
    ) -> Optional[str]:
        """
        Generate audio from text.
        
        Args:
            text: Text to convert to speech
            output_path: Where to save the audio file (optional)
            voice: Voice to use (optional, uses config default)
            speed: Speech speed (optional, uses config default)
            
        Returns:
            Path to generated audio file, or None if failed
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # Use provided parameters or fall back to config
            voice = voice or self.config.voice
            speed = speed or self.config.speed
            
            # Prepare text
            clean_text = self._prepare_text(text)
            logger.info(f"Generating TTS for {len(clean_text)} characters with voice '{voice}'")
            
            # Split text if too long
            text_chunks = self._split_text_intelligently(clean_text)
            logger.debug(f"Split text into {len(text_chunks)} chunks")
            
            all_audio = []
            
            # Process each chunk
            for i, chunk in enumerate(text_chunks):
                logger.debug(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
                
                generator = self.pipeline(
                    chunk,
                    voice=voice,
                    speed=speed,
                    split_pattern=self.config.split_pattern
                )
                
                chunk_audio = []
                for j, (gs, ps, audio) in enumerate(generator):
                    chunk_audio.append(audio)
                    logger.debug(f"  Segment {j+1}: {len(audio)} samples")
                
                # Concatenate chunk segments
                if chunk_audio:
                    import numpy as np
                    chunk_combined = np.concatenate(chunk_audio)
                    all_audio.append(chunk_combined)
                    
                    # Add pause between chunks
                    if i < len(text_chunks) - 1:  # Not the last chunk
                        pause_samples = int(0.5 * self.config.sample_rate)  # 0.5 second pause
                        pause = np.zeros(pause_samples, dtype=chunk_combined.dtype)
                        all_audio.append(pause)
            
            if not all_audio:
                logger.error("No audio generated")
                return None
            
            # Combine all audio
            import numpy as np
            final_audio = np.concatenate(all_audio)
            
            # Add padding if requested
            if self.config.add_silence_padding:
                padding_samples = int(self.config.padding_seconds * self.config.sample_rate)
                padding = np.zeros(padding_samples, dtype=final_audio.dtype)
                final_audio = np.concatenate([padding, final_audio, padding])
            
            # Normalize audio if requested
            if self.config.normalize_audio:
                max_val = np.abs(final_audio).max()
                if max_val > 0:
                    final_audio = final_audio / max_val * 0.9  # Leave some headroom
            
            # Determine output path
            if output_path is None:
                timestamp = int(time.time())
                output_path = f"tts_output_{timestamp}.{self.config.output_format}"
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save audio file
            sf.write(str(output_path), final_audio, self.config.sample_rate)
            
            generation_time = time.time() - start_time
            duration = len(final_audio) / self.config.sample_rate
            
            logger.success(f"Generated {duration:.1f}s audio in {generation_time:.2f}s")
            logger.info(f"Audio saved to: {output_path}")
            logger.info(f"Real-time factor: {duration/generation_time:.1f}x")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None
    
    async def generate_from_article(
        self, 
        article: Article, 
        output_dir: Optional[str] = None,
        voice: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate TTS audio from a summarized article.
        
        Args:
            article: Article object with content/summary
            output_dir: Directory to save audio files
            voice: Voice to use for generation
            
        Returns:
            Path to generated audio file
        """
        try:
            # Use article content or summary
            text = getattr(article, 'tiktok_summary', None) or article.content
            if not text:
                logger.error("No text content found in article")
                return None
            
            # Create output filename
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Sanitize title for filename
                safe_title = "".join(c for c in article.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_title = safe_title[:50]  # Limit length
                
                output_path = output_dir / f"{safe_title}_{int(time.time())}.{self.config.output_format}"
            else:
                output_path = None
            
            return await self.generate_audio(text, str(output_path) if output_path else None, voice)
            
        except Exception as e:
            logger.error(f"Failed to generate TTS from article: {e}")
            return None
    
    async def batch_generate(
        self, 
        texts: List[str], 
        output_dir: str,
        voice: Optional[str] = None,
        filename_prefix: str = "tts_batch"
    ) -> List[Optional[str]]:
        """
        Generate TTS for multiple texts in batch.
        
        Args:
            texts: List of texts to convert
            output_dir: Directory to save audio files
            voice: Voice to use for all generations
            filename_prefix: Prefix for output filenames
            
        Returns:
            List of output file paths (None for failed generations)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Processing batch item {i+1}/{len(texts)}")
            
            output_path = output_dir / f"{filename_prefix}_{i+1:03d}.{self.config.output_format}"
            result = await self.generate_audio(text, str(output_path), voice)
            results.append(result)
        
        successful = sum(1 for r in results if r is not None)
        logger.info(f"Batch complete: {successful}/{len(texts)} successful")
        
        return results
    
    def get_audio_info(self, audio_path: str) -> Dict:
        """Get information about an audio file."""
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                return {"error": "File not found"}
            
            # Load audio to get info
            audio_data, sample_rate = sf.read(str(audio_path))
            
            duration = len(audio_data) / sample_rate
            file_size = audio_path.stat().st_size
            
            return {
                "path": str(audio_path),
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": audio_data.ndim,
                "samples": len(audio_data),
                "file_size": file_size,
                "format": audio_path.suffix[1:],
                "bitrate": file_size * 8 / duration if duration > 0 else 0
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def cleanup(self):
        """Clean up resources."""
        if self.pipeline:
            # Kokoro doesn't need explicit cleanup, but we can clear references
            self.pipeline = None
        self.initialized = False
        logger.info("Kokoro TTS engine cleaned up")


# Convenience functions
async def create_tts_engine(config: TTSConfig = None) -> KokoroTTSEngine:
    """Create and initialize a TTS engine."""
    engine = KokoroTTSEngine(config)
    await engine.initialize()
    return engine


async def quick_tts(
    text: str, 
    output_path: str = None, 
    voice: str = 'af_heart',
    speed: float = 1.0
) -> Optional[str]:
    """Quick TTS generation for simple use cases."""
    config = TTSConfig(voice=voice, speed=speed)
    engine = KokoroTTSEngine(config)
    
    try:
        await engine.initialize()
        result = await engine.generate_audio(text, output_path)
        await engine.cleanup()
        return result
    except Exception as e:
        logger.error(f"Quick TTS failed: {e}")
        await engine.cleanup()
        return None


if __name__ == "__main__":
    # Example usage
    async def main():
        # Test basic functionality
        config = TTSConfig(voice='af_heart', speed=1.1)
        tts = KokoroTTSEngine(config)
        
        await tts.initialize()
        
        test_text = """
        This is a test of the Kokoro TTS engine for TikTok Automata! 
        It should generate high-quality speech that sounds natural and engaging.
        Perfect for creating awesome TikTok content!
        """
        
        output_file = await tts.generate_audio(test_text, "test_output.wav")
        
        if output_file:
            info = tts.get_audio_info(output_file)
            print(f"Generated: {info}")
        
        await tts.cleanup()
    
    # Run the test
    asyncio.run(main())
