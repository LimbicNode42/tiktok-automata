#!/usr/bin/env python3
"""
TikTok Automata Daily Scheduler
==============================

Simple scheduler to run the production pipeline at a specified time each day.
Can be used as an alternative to cron jobs.

Usage:
    python daily_scheduler.py                    # Run at 9:00 AM daily
    python daily_scheduler.py --time 14:30       # Run at 2:30 PM daily
    python daily_scheduler.py --once             # Run once now and exit
"""

import asyncio
import logging
import schedule
import time
import sys
import argparse
from datetime import datetime
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_scheduler.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TikTokScheduler")

class DailyScheduler:
    """Daily scheduler for TikTok Automata production pipeline."""
    
    def __init__(self):
        self.running = True
        
    def run_production_pipeline(self):
        """Execute the production pipeline."""
        logger.info("Starting scheduled production pipeline run")
        
        try:
            # Check if production pipeline exists
            if not Path("production_pipeline.py").exists():
                logger.error("production_pipeline.py not found in current directory")
                return
            
            # Run the production pipeline
            result = subprocess.run(
                [sys.executable, "production_pipeline.py"],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("Production pipeline completed successfully")
                logger.info(f"Output: {result.stdout}")
            else:
                logger.error(f"Production pipeline failed with code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("Production pipeline timed out after 1 hour")
        except Exception as e:
            logger.error(f"Error running production pipeline: {e}")
    
    def start_scheduler(self, run_time: str = "09:00"):
        """Start the daily scheduler."""
        logger.info(f"Starting TikTok Automata daily scheduler")
        logger.info(f"Scheduled to run daily at {run_time}")
        
        # Schedule the job
        schedule.every().day.at(run_time).do(self.run_production_pipeline)
        
        # Log next run time
        next_run = schedule.next_run()
        logger.info(f"Next scheduled run: {next_run}")
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
    
    def run_once(self):
        """Run the pipeline once and exit."""
        logger.info("Running production pipeline once")
        self.run_production_pipeline()
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False

def main():
    """Main entry point for the scheduler."""
    parser = argparse.ArgumentParser(description="TikTok Automata Daily Scheduler")
    parser.add_argument('--time', type=str, default="09:00",
                       help='Time to run daily (HH:MM format, default: 09:00)')
    parser.add_argument('--once', action='store_true',
                       help='Run once immediately and exit')
    
    args = parser.parse_args()
    
    # Validate time format
    if not args.once:
        try:
            time_obj = datetime.strptime(args.time, "%H:%M")
        except ValueError:
            logger.error(f"Invalid time format: {args.time}. Use HH:MM format (e.g., 14:30)")
            sys.exit(1)
    
    scheduler = DailyScheduler()
    
    if args.once:
        scheduler.run_once()
    else:
        try:
            scheduler.start_scheduler(args.time)
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user")
        except Exception as e:
            logger.error(f"Scheduler failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
