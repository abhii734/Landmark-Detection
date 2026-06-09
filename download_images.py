"""
Landmark Image Downloader

USAGE:
    python download_images.py                    # Download all images
    python download_images.py --limit 100        # Download only first 100
    python download_images.py --sample 0.01      # Download 1% sample
    python download_images.py --resume           # Resume from last checkpoint
    python download_images.py --force            # Re-download existing files
    python download_images.py --workers 100     # Adjust concurrency

This script downloads landmark images from URLs in train.csv with the following features:
    - Async parallel downloads (default: 50 concurrent workers)
    - Retry logic (3 attempts per image)
    - Resume capability (saves downloaded IDs to checkpoint file)
    - Progress bar with ETA and speed stats
    - Skips failed downloads gracefully
    - Saves images to images/{id}.jpg format
"""

import argparse
import asyncio
import aiohttp
import aiofiles
import csv
import os
import hashlib
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Configuration
DEFAULT_CSV_PATH = Path(__file__).parent / "train.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "images"
DEFAULT_CHECKPOINT_FILE = Path(__file__).parent / ".download_checkpoint.txt"
MAX_CONCURRENT_DOWNLOADS = 50
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
REQUEST_TIMEOUT = 30  # seconds
CHUNK_SIZE = 8192


class DownloadProgress:
    """Tracks download progress for resume capability."""
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.downloaded_ids: set[str] = set()
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load previously downloaded IDs from checkpoint file."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                self.downloaded_ids = set(line.strip() for line in f if line.strip())
    
    def save_checkpoint(self, image_id: str):
        """Save a successfully downloaded image ID."""
        self.downloaded_ids.add(image_id)
        with open(self.checkpoint_file, 'a') as f:
            f.write(f"{image_id}\n")
    
    def is_downloaded(self, image_id: str) -> bool:
        """Check if image was already downloaded."""
        return image_id in self.downloaded_ids


async def download_image(
    session: aiohttp.ClientSession,
    image_id: str,
    url: str,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    progress: DownloadProgress,
    force: bool = False
) -> tuple[str, bool]:
    """
    Download a single image with retry logic.
    
    Returns:
        tuple: (image_id, success_flag)
    """
    output_path = output_dir / f"{image_id}.jpg"
    
    # Skip if already downloaded (unless force is True)
    if not force and progress.is_downloaded(image_id):
        return image_id, True
    
    # Skip if file already exists
    if not force and output_path.exists():
        progress.save_checkpoint(image_id)
        return image_id, True
    
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as response:
                    if response.status == 404:
                        # Image not found, skip without retry
                        return image_id, False
                    
                    if response.status != 200:
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status
                        )
                    
                    # Check content type
                    content_type = response.headers.get('Content-Type', '')
                    if 'image' not in content_type.lower() and 'jpeg' not in content_type.lower():
                        # Try anyway - some servers don't set proper content type
                        pass
                    
                    # Download content
                    content = await response.read()
                    
                    # Validate it's actually an image (basic check)
                    if len(content) < 1000:  # Too small to be a real image
                        raise ValueError(f"File too small: {len(content)} bytes")
                    
                    # Save to disk
                    async with aiofiles.open(output_path, 'wb') as f:
                        await f.write(content)
                    
                    progress.save_checkpoint(image_id)
                    return image_id, True
                    
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    # Final attempt failed
                    if output_path.exists():
                        try:
                            output_path.unlink()
                        except OSError:
                            pass
                    return image_id, False
        
        return image_id, False


async def download_batch(
    batch: list[tuple[str, str]],
    output_dir: Path,
    progress: DownloadProgress,
    force: bool = False,
    total: int = 0
) -> tuple[int, int]:
    """
    Download a batch of images concurrently.
    
    Returns:
        tuple: (success_count, failure_count)
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS, ttl_dns_cache=300)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            download_image(session, image_id, url, output_dir, semaphore, progress, force)
            for image_id, url in batch
        ]
        
        results = []
        with tqdm(total=len(batch), desc="Downloading", unit="img") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
        
        success = sum(1 for _, ok in results if ok)
        failed = len(results) - success
        return success, failed


def read_csv(csv_path: Path, limit: Optional[int] = None, sample_rate: Optional[float] = None) -> list[tuple[str, str]]:
    """
    Read image URLs from CSV file or its split parts.
    
    Args:
        csv_path: Path to train.csv
        limit: Maximum number of rows to read
        sample_rate: Fraction of rows to sample (0.0 to 1.0)
    
    Returns:
        list of (image_id, url) tuples
    """
    results = []
    csv_path = Path(csv_path)
    
    csv_files = []
    if csv_path.exists():
        csv_files = [csv_path]
    else:
        parent = csv_path.parent
        name = csv_path.stem
        ext = csv_path.suffix
        csv_files = sorted(parent.glob(f"{name}_part_*{ext}"))
        if not csv_files:
            csv_files = sorted(parent.glob(f"{name}_*{ext}"))
            
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching {csv_path}")
        
    print(f"Reading from {len(csv_files)} CSV files...")
    
    total_read = 0
    for file_path in csv_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if limit and len(results) >= limit:
                    break
                
                if sample_rate and (total_read % int(1 / sample_rate) != 0):
                    total_read += 1
                    continue
                
                results.append((row['id'], row['url']))
                total_read += 1
                
            if limit and len(results) >= limit:
                break
                
    return results


def create_sample_csv(csv_path: Path, sample_rate: float, sample_name: str = "sample"):
    """Create a smaller sample CSV for testing."""
    output_path = csv_path.parent / f"{sample_name}.csv"
    csv_path = Path(csv_path)
    
    csv_files = []
    if csv_path.exists():
        csv_files = [csv_path]
    else:
        parent = csv_path.parent
        name = csv_path.stem
        ext = csv_path.suffix
        csv_files = sorted(parent.glob(f"{name}_part_*{ext}"))
        if not csv_files:
            csv_files = sorted(parent.glob(f"{name}_*{ext}"))
            
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching {csv_path}")
        
    print(f"Reading from {len(csv_files)} CSV files...")
    
    total = 0
    for file_path in csv_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            total += sum(1 for _ in f) - 1
            
    sample_size = int(total * sample_rate)
    rows = []
    
    for file_path in csv_files:
        with open(file_path, 'r', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                if len(rows) >= sample_size:
                    break
                rows.append(row)
            if len(rows) >= sample_size:
                break
                
    with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=['id', 'url', 'landmark_id'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Created {output_path} with {len(rows)} rows")
    exit()


async def main(args):
    """Main downloader function."""
    csv_path = Path(args.csv) if args.csv else DEFAULT_CSV_PATH
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    checkpoint_file = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT_FILE
    
    # Validate CSV exists
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV
    print(f"Reading {csv_path}...")
    images = read_csv(csv_path, limit=args.limit, sample_rate=args.sample)
    
    if not images:
        print("No images to download.")
        return
    
    # Filter based on resume option
    progress = DownloadProgress(checkpoint_file)
    
    if not args.resume and not args.force:
        to_download = [(id_, url) for id_, url in images if not progress.is_downloaded(id_)]
        skipped = len(images) - len(to_download)
        print(f"Total images: {len(images)}, Already downloaded: {skipped}, To download: {len(to_download)}")
    elif args.force:
        to_download = images
        print(f"Force mode: Re-downloading all {len(images)} images")
    else:
        to_download = [(id_, url) for id_, url in images]
        print(f"Resume mode: {len(to_download)} images to process")
    
    if not to_download:
        print("No images to download. All complete!")
        return
    
    # Process in batches to show progress
    batch_size = args.batch_size
    total_success = 0
    total_failed = 0
    
    for i in range(0, len(to_download), batch_size):
        batch = to_download[i:i + batch_size]
        start_idx = i + 1
        print(f"\nBatch {i // batch_size + 1}: Processing images {start_idx}-{start_idx + len(batch) - 1}")
        
        success, failed = await download_batch(
            batch, output_dir, progress, 
            force=args.force, total=len(to_download)
        )
        total_success += success
        total_failed += failed
        
        print(f"Batch complete: {success} success, {failed} failed")
    
    print(f"\n{'='*50}")
    print(f"Download Summary:")
    print(f"  Total processed: {total_success + total_failed}")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {total_failed}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*50}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download landmark images from CSV URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_images.py                      # Download all images
  python download_images.py --limit 100          # Download first 100 only
  python download_images.py --sample 0.01       # Download 1%% sample
  python download_images.py --resume             # Resume interrupted download
  python download_images.py --force             # Re-download existing files
  python download_images.py --workers 100       # Increase concurrent workers
        """
    )
    
    parser.add_argument(
        '--csv', '-c',
        type=str,
        default=None,
        help=f"Path to CSV file (default: {DEFAULT_CSV_PATH})"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        '--checkpoint', '-p',
        type=str,
        default=None,
        help=f"Checkpoint file for resume (default: {DEFAULT_CHECKPOINT_FILE})"
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help="Limit number of images to download"
    )
    
    parser.add_argument(
        '--sample', '-s',
        type=float,
        default=None,
        help="Download a sample fraction (0.0 to 1.0)"
    )
    
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help="Resume from last checkpoint, skipping already downloaded images"
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help="Force re-download even if files exist"
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=MAX_CONCURRENT_DOWNLOADS,
        help=f"Number of concurrent downloads (default: {MAX_CONCURRENT_DOWNLOADS})"
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=500,
        help="Process images in batches (default: 500)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Override global concurrency setting
    MAX_CONCURRENT_DOWNLOADS = args.workers
    
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n\nDownload interrupted. Use --resume to continue.")
        exit(1)