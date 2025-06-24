"""
Configuration utilities for TLDR Newsletter Scraper.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ScrapingConfig:
    """Scraping configuration settings."""
    max_age_hours: int = 24
    max_articles_per_run: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


@dataclass
class VoiceProfile:
    """Voice profile for TTS generation."""
    id: str
    name: str
    description: str
    gender: str
    accent: str
    personality: str
    best_for: List[str]


@dataclass
class TTSConfig:
    """TTS configuration settings."""
    default_voice: str = 'af_heart'
    default_speed: float = 1.55  # Optimized for TikTok/short-form content - Natural 1.55x speed
    sample_rate: int = 24000
    output_format: str = 'wav'
    normalize_audio: bool = True
    add_silence_padding: bool = True
    padding_seconds: float = 0.5
    target_duration: float = 60.0


class Config:
    """Central configuration manager."""
    
    def __init__(self):
        self.scraping = ScrapingConfig(
            max_age_hours=int(os.getenv("MAX_AGE_HOURS", 24)),
            max_articles_per_run=int(os.getenv("MAX_ARTICLES_PER_RUN", 10)),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", 30)),
            retry_attempts=int(os.getenv("RETRY_ATTEMPTS", 3)),
            user_agent=os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        )        # TTS Configuration
        self.tts = TTSConfig(
            default_voice=os.getenv("TTS_DEFAULT_VOICE", "af_heart"),
            default_speed=float(os.getenv("TTS_DEFAULT_SPEED", 1.55)),  # Natural 1.55x for TikTok
            sample_rate=int(os.getenv("TTS_SAMPLE_RATE", 24000)),
            output_format=os.getenv("TTS_OUTPUT_FORMAT", "wav"),
            normalize_audio=os.getenv("TTS_NORMALIZE_AUDIO", "true").lower() == "true",
            add_silence_padding=os.getenv("TTS_ADD_PADDING", "true").lower() == "true",
            padding_seconds=float(os.getenv("TTS_PADDING_SECONDS", 0.5)),
            target_duration=float(os.getenv("TTS_TARGET_DURATION", 60.0))
        )
        
        # Kokoro Voice Profiles
        self.voice_profiles = self._initialize_voice_profiles()
        
        # API endpoints
        self.tldr_rss_url = os.getenv("TLDR_RSS_FEED", "https://tldr.tech/rss")
        self.tldr_base_url = os.getenv("TLDR_NEWSLETTER_URL", "https://tldr.tech/")
        
        # Paths
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "./data"))
        self.logs_dir = Path(os.getenv("LOGS_DIR", "./logs"))
          # Ensure directories exist        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_voice_profiles(self) -> Dict[str, VoiceProfile]:
        """Initialize Kokoro voice profiles with detailed configurations."""
        return {
            # American Female Voices
            'af_heart': VoiceProfile(
                id='af_heart',
                name='Heart',
                description='Warm, emotional female voice',
                gender='female',
                accent='american',
                personality='warm, emotional, engaging',
                best_for=['lifestyle', 'personal stories', 'emotional content', 'general TikTok']
            ),
            'af_alloy': VoiceProfile(
                id='af_alloy',
                name='Alloy',
                description='Professional female voice',
                gender='female',
                accent='american',
                personality='professional, clear, reliable',
                best_for=['business', 'educational content', 'presentations', 'formal announcements']
            ),
            'af_aoede': VoiceProfile(
                id='af_aoede',
                name='Aoede',
                description='Artistic female voice',
                gender='female',
                accent='american',
                personality='artistic, expressive, creative',
                best_for=['arts', 'creative content', 'storytelling', 'cultural topics']
            ),
            'af_bella': VoiceProfile(
                id='af_bella',
                name='Bella',
                description='High-energy female voice',
                gender='female',
                accent='american',
                personality='energetic, dynamic, engaging',
                best_for=['entertainment', 'high-energy content', 'fitness', 'motivational']
            ),
            'af_jessica': VoiceProfile(
                id='af_jessica',
                name='Jessica',
                description='Casual female voice',
                gender='female',
                accent='american',
                personality='casual, friendly, approachable',
                best_for=['casual content', 'everyday topics', 'friendly conversations']
            ),
            'af_kore': VoiceProfile(
                id='af_kore',
                name='Kore',
                description='Balanced female voice',
                gender='female',
                accent='american',
                personality='balanced, versatile, clear',
                best_for=['general content', 'news', 'educational', 'versatile use']
            ),
            'af_nicole': VoiceProfile(
                id='af_nicole',
                name='Nicole',
                description='Tech-focused female voice',
                gender='female',
                accent='american',
                personality='tech-savvy, professional, modern',
                best_for=['tech content', 'gaming', 'digital trends', 'modern topics']
            ),
            'af_nova': VoiceProfile(
                id='af_nova',
                name='Nova',
                description='Fresh female voice',
                gender='female',
                accent='american',
                personality='fresh, youthful, energetic',
                best_for=['youth content', 'trends', 'pop culture', 'fresh perspectives']
            ),
            'af_river': VoiceProfile(
                id='af_river',
                name='River',
                description='Flowing female voice',
                gender='female',
                accent='american',
                personality='smooth, flowing, natural',
                best_for=['nature content', 'calm topics', 'meditation', 'wellness']
            ),
            'af_sarah': VoiceProfile(
                id='af_sarah',
                name='Sarah',
                description='Reliable female voice',
                gender='female',
                accent='american',
                personality='reliable, trustworthy, consistent',
                best_for=['news', 'information', 'reliable content', 'updates']
            ),
            'af_sky': VoiceProfile(
                id='af_sky',
                name='Sky',
                description='Clear, professional female voice',
                gender='female',
                accent='american',
                personality='professional, clear, authoritative',
                best_for=['news', 'business', 'educational content', 'tech reviews']
            ),
            
            # American Male Voices
            'am_adam': VoiceProfile(
                id='am_adam',
                name='Adam',
                description='Strong, confident male voice',
                gender='male',
                accent='american',
                personality='confident, strong, assertive',
                best_for=['sports', 'action content', 'gaming', 'motivational']
            ),
            'am_echo': VoiceProfile(
                id='am_echo',
                name='Echo',
                description='Resonant male voice',
                gender='male',
                accent='american',
                personality='resonant, deep, impactful',
                best_for=['dramatic content', 'storytelling', 'announcements', 'deep topics']
            ),
            'am_eric': VoiceProfile(
                id='am_eric',
                name='Eric',
                description='Friendly male voice',
                gender='male',
                accent='american',
                personality='friendly, approachable, warm',
                best_for=['casual content', 'friendly conversations', 'everyday topics', 'approachable content']
            ),
            'am_fenrir': VoiceProfile(
                id='am_fenrir',
                name='Fenrir',
                description='Powerful male voice',
                gender='male',
                accent='american',
                personality='powerful, commanding, strong',
                best_for=['epic content', 'gaming', 'fantasy', 'powerful narratives']
            ),
            'am_liam': VoiceProfile(
                id='am_liam',
                name='Liam',
                description='Casual male voice',
                gender='male',
                accent='american',
                personality='casual, laid-back, relatable',
                best_for=['casual content', 'commentary', 'everyday observations', 'relaxed topics']
            ),
            'am_michael': VoiceProfile(
                id='am_michael',
                name='Michael',
                description='Professional, news-anchor male voice',
                gender='male',
                accent='american',
                personality='professional, authoritative, trustworthy',
                best_for=['news', 'documentary style', 'serious topics', 'analysis']
            ),
            'am_onyx': VoiceProfile(
                id='am_onyx',
                name='Onyx',
                description='Sophisticated male voice',
                gender='male',
                accent='american',
                personality='sophisticated, refined, elegant',
                best_for=['luxury content', 'sophisticated topics', 'refined discussions', 'premium brands']
            ),
            'am_puck': VoiceProfile(
                id='am_puck',
                name='Puck',
                description='Playful male voice',
                gender='male',
                accent='american',
                personality='playful, mischievous, entertaining',
                best_for=['entertainment', 'humor', 'playful content', 'comedy']
            ),
            'am_santa': VoiceProfile(
                id='am_santa',
                name='Santa',
                description='Jolly male voice',
                gender='male',
                accent='american',
                personality='jolly, warm, festive',
                best_for=['holiday content', 'festive topics', 'warm content', 'seasonal']
            ),
            
            # British Female Voices
            'bf_alice': VoiceProfile(
                id='bf_alice',
                name='Alice',
                description='Classic British female voice',
                gender='female',
                accent='british',
                personality='classic, proper, traditional',
                best_for=['traditional content', 'classic topics', 'proper discussions', 'heritage']
            ),
            'bf_emma': VoiceProfile(
                id='bf_emma',
                name='Emma',
                description='British female voice',
                gender='female',
                accent='british',
                personality='sophisticated, articulate, elegant',
                best_for=['luxury content', 'cultural topics', 'sophisticated humor', 'premium brands']
            ),
            'bf_isabella': VoiceProfile(
                id='bf_isabella',
                name='Isabella',
                description='Elegant British female voice',
                gender='female',
                accent='british',
                personality='refined, elegant, cultured',
                best_for=['fashion', 'arts', 'literature', 'high-end content']
            ),
            'bf_lily': VoiceProfile(
                id='bf_lily',
                name='Lily',
                description='Sweet British female voice',
                gender='female',
                accent='british',
                personality='sweet, gentle, charming',
                best_for=['gentle content', 'sweet topics', 'charming discussions', 'soft content']
            ),
            
            # British Male Voices
            'bm_daniel': VoiceProfile(
                id='bm_daniel',
                name='Daniel',
                description='Professional British male voice',
                gender='male',
                accent='british',
                personality='professional, reliable, steady',
                best_for=['professional content', 'business topics', 'reliable information', 'steady delivery']
            ),
            'bm_fable': VoiceProfile(
                id='bm_fable',
                name='Fable',
                description='Storytelling British male voice',
                gender='male',
                accent='british',
                personality='storytelling, narrative, engaging',
                best_for=['storytelling', 'narratives', 'fairy tales', 'engaging content']
            ),
            'bm_george': VoiceProfile(
                id='bm_george',
                name='George',
                description='Distinguished British male voice',
                gender='male',
                accent='british',
                personality='distinguished, authoritative, scholarly',
                best_for=['educational content', 'history', 'science', 'formal topics']
            ),            'bm_lewis': VoiceProfile(
                id='bm_lewis',
                name='Lewis',
                description='Casual British male voice',
                gender='male',
                accent='british',
                personality='casual, friendly, approachable',
                best_for=['casual content', 'humor', 'everyday topics', 'commentary']
            )
        }
    
    def get_voice_profile(self, voice_id: str) -> VoiceProfile:
        """Get a specific voice profile by ID."""
        return self.voice_profiles.get(voice_id)
    
    def get_voices_by_gender(self, gender: str) -> List[VoiceProfile]:
        """Get all voice profiles of a specific gender."""
        return [profile for profile in self.voice_profiles.values() if profile.gender == gender]
    
    def get_voices_by_accent(self, accent: str) -> List[VoiceProfile]:
        """Get all voice profiles with a specific accent."""
        return [profile for profile in self.voice_profiles.values() if profile.accent == accent]
    
    def get_recommended_voice(self, content_type: str) -> VoiceProfile:
        """Get recommended voice for specific content type."""
        # Define content type to voice mappings based on quality grades
        content_mappings = {
            'ai': 'af_bella',           # High-energy female for AI content
            'tech': 'af_nicole',        # Tech-focused female for tech news
            'business': 'af_alloy',     # Professional female for business
            'science': 'bm_george',     # Distinguished British for science
            'gaming': 'af_bella',       # High-energy for gaming
            'lifestyle': 'af_heart',    # Warm female for lifestyle
            'news': 'am_michael',       # News anchor voice for breaking news
            'entertainment': 'am_puck', # Playful male for entertainment
            'casual': 'bm_lewis',       # Casual British for everyday content
            'motivational': 'af_bella', # High-energy voice for motivation
            'comedy': 'am_puck',        # Playful voice for comedy
            'education': 'bm_george',   # Scholarly voice for education
            'wellness': 'af_river',     # Flowing voice for wellness
            'storytelling': 'bm_fable'  # Storytelling voice for narratives
        }
        
        voice_id = content_mappings.get(content_type.lower(), self.tts.default_voice)
        return self.get_voice_profile(voice_id)
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for scraping requests."""
        return {
            'User-Agent': self.scraping.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'scraping': {
                'max_age_hours': self.scraping.max_age_hours,
                'max_articles_per_run': self.scraping.max_articles_per_run,
                'request_timeout': self.scraping.request_timeout,
                'retry_attempts': self.scraping.retry_attempts,
                'user_agent': self.scraping.user_agent
            },
            'tts': {
                'default_voice': self.tts.default_voice,
                'default_speed': self.tts.default_speed,
                'sample_rate': self.tts.sample_rate,
                'output_format': self.tts.output_format,
                'normalize_audio': self.tts.normalize_audio,
                'add_silence_padding': self.tts.add_silence_padding,
                'padding_seconds': self.tts.padding_seconds,
                'target_duration': self.tts.target_duration
            },
            'voice_profiles': {
                voice_id: {
                    'id': profile.id,
                    'name': profile.name,
                    'description': profile.description,
                    'gender': profile.gender,
                    'accent': profile.accent,
                    'personality': profile.personality,
                    'best_for': profile.best_for
                }
                for voice_id, profile in self.voice_profiles.items()
            },
            'urls': {
                'tldr_rss': self.tldr_rss_url,
                'tldr_base': self.tldr_base_url
            },
            'paths': {
                'output_dir': str(self.output_dir),
                'logs_dir': str(self.logs_dir)
            }
        }


# Global config instance
config = Config()

if __name__ == "__main__":
    print("Config loaded successfully!")
    print(f"RSS URL: {config.tldr_rss_url}")
    print(f"Output dir: {config.output_dir}")
    
    # Display TTS configuration
    print(f"\n🎤 TTS Configuration:")
    print(f"  Default voice: {config.tts.default_voice}")
    print(f"  Default speed: {config.tts.default_speed}x")
    print(f"  Sample rate: {config.tts.sample_rate} Hz")
    print(f"  Target duration: {config.tts.target_duration}s")
    
    # Display available voices
    print(f"\n🎭 Available Voices ({len(config.voice_profiles)}):")
    for voice_id, profile in config.voice_profiles.items():
        print(f"  • {profile.name} ({voice_id}): {profile.description}")
        print(f"    Gender: {profile.gender.title()}, Accent: {profile.accent.title()}")
        print(f"    Best for: {', '.join(profile.best_for)}")
        print()
    
    # Show content type recommendations
    print("🎯 Content Type Recommendations:")
    content_types = ['ai', 'tech', 'business', 'science', 'gaming', 'lifestyle']
    for content_type in content_types:
        recommended = config.get_recommended_voice(content_type)
        print(f"  {content_type.title()}: {recommended.name} ({recommended.id}) - {recommended.description}")
    
    # Show voices by gender
    print(f"\n👩 Female Voices: {len(config.get_voices_by_gender('female'))}")
    for profile in config.get_voices_by_gender('female'):
        print(f"  • {profile.name} ({profile.accent}): {profile.description}")
    
    print(f"\n👨 Male Voices: {len(config.get_voices_by_gender('male'))}")
    for profile in config.get_voices_by_gender('male'):
        print(f"  • {profile.name} ({profile.accent}): {profile.description}")
