# Legacy Data Directory

This directory is kept for backward compatibility but is no longer used for output files.

## New File Locations:
- **Scraper outputs**: `src/scraper/data/`
  - `extraction_patterns.json` - Learned extraction patterns
  - `tldr_articles_*.json` - Extracted newsletter articles

- **Summarizer outputs**: `src/summarizer/data/`
  - `tiktok_summaries_*.json` - Batch TikTok summaries
  - `tiktok_summary_*.json` - Individual TikTok summaries

## Migration Complete
All data files have been moved to their respective module directories to improve organization and maintainability.
