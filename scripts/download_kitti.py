"""Download KITTI dataset for object detection.

This script downloads the KITTI object detection dataset.
Based on the CMPE_KITTI.ipynb notebook.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm


# KITTI object detection training set URLs
KITTI_URLS = {
    "images": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "labels": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
    "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
}


class DownloadProgressBar(tqdm):
    """Progress bar for download."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> None:
    """Download a file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download KITTI object detection dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./kitti_data",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--no-velodyne",
        action="store_true",
        help="Skip downloading velodyne data (LiDAR)",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("KITTI Object Detection Dataset Downloader")
    print("-" * 50)
    print(f"Output directory: {args.output_dir}")
    print(f"Skip existing: {args.skip_existing}")
    print("-" * 50)
    print()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output_dir)
    
    # Download each component
    for name, url in KITTI_URLS.items():
        filename = url.split("/")[-1]
        filepath = Path(filename)
        
        # Download
        if not filepath.exists():
            print(f"Downloading {name}...")
            try:
                download_file(url, filepath)
            except Exception as e:
                print(f"Error downloading {name}: {e}")
                continue
        
        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall()
            print(f"Extracted {name}")
        except Exception as e:
            print(f"Error extracting {name}: {e}")
        
        print()
    
    print("✅ Download complete!")


if __name__ == "__main__":
    main()

