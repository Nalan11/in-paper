import os
import time
import logging
import shutil
import asyncio

logger = logging.getLogger(__name__)

def cleanup_old_files(directories, max_age_seconds=3600):
    """
    Cleans up files and folders in the given directories that are older than max_age_seconds.
    """
    now = time.time()
    cleaned_count = 0
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                # Check modification time
                mtime = os.path.getmtime(file_path)
                if now - mtime > max_age_seconds:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    cleaned_count += 1
                    logger.info(f"Cleaned up old file/dir: {file_path}")
            except Exception as e:
                logger.error(f"Failed to clean up {file_path}: {e}")
                
    return cleaned_count

async def background_cleanup_task(directories, interval_seconds=3600, max_age_seconds=3600):
    """
    An asyncio task that periodically cleans up old files to prevent disk full issues.
    """
    while True:
        try:
            cleaned = cleanup_old_files(directories, max_age_seconds)
            if cleaned > 0:
                print(f"Cleanup Task: Removed {cleaned} old files/directories.")
        except Exception as e:
            logger.error(f"Error in background cleanup task: {e}")
        
        await asyncio.sleep(interval_seconds)
