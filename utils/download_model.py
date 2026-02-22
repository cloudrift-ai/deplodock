import os
import time
from huggingface_hub import snapshot_download
from argparse import ArgumentParser
from tqdm.auto import tqdm


def download_and_track(repo_id, local_dir=None):
    """
    Downloads a Hugging Face model and tracks its average download speed.

    Args:
        repo_id (str): The ID of the model repository on Hugging Face (e.g., "bert-base-uncased").
        local_dir (str, optional): The local directory to save the model to.
                                   If None, it will be saved in the Hugging Face cache.

    Note:
        Set the HF_HOME environment variable before calling this function to control
        where HuggingFace stores all cache data (tokens, temp files, etc.)
    """

    # Enable hf_transfer for potentially faster downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print(f"Downloading model: {repo_id}...")

    # Get information about the repository to track total size for speed calculation
    # (Optional: you might need to adjust this depending on if you want to track a single file or multiple)
    # The snapshot_download function automatically handles progress bars internally,
    # but we can wrap it with tqdm for overall progress and custom speed tracking.

    start_time = time.time()

    # If `local_dir` is not specified, it will download to the Hugging Face cache (~/.cache/huggingface/hub)
    # The snapshot_download function returns the path to the downloaded repository.

    # For a more advanced approach to download and track, you could potentially get
    # the total size and download chunks manually, calculating speed per chunk,
    # but snapshot_download abstracts that, and relying on its internal progress
    # combined with overall time will give you an average speed.

    download_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        # You can add a progress_callback to snapshot_download for more granular control,
        # but for simplicity, we'll track the overall download time for average speed.
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Model download complete! Saved to: {download_path}")

    # To calculate average speed, you would ideally know the total size of the downloaded files.
    # Since snapshot_download handles multiple files, getting the precise total size beforehand
    # might require inspecting the repo on Hugging Face. For a simple average, we can approximate
    # if we know the typical size or get it after the fact.
    # For now, we'll just report the time taken.
    print(f"Download took {elapsed_time:.2f} seconds.")

    # You could potentially iterate through the downloaded files in 'download_path'
    # and calculate the total size to get a more accurate average speed in MB/s or MiB/s.
    # For example:
    total_size = 0
    for root, _, files in os.walk(download_path):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))

    if total_size > 0:
        average_speed_bps = total_size / elapsed_time
        average_speed_mbps = average_speed_bps / (1024 * 1024) * 8  # Bytes per second to MiB per second
        print(f"Average download speed: {average_speed_mbps:.2f} MiB/s")
    else:
        print("Could not determine total size for average speed calculation.")


# Example usage:
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model-name', default="meta-llama/Llama-3.1-8B-Instruct",
                        help='HuggingFace model repository ID')
    parser.add_argument("--hg-dir", default="../models",
                        help='Directory to download the model to')
    args = parser.parse_args()

    download_and_track(args.model_name, local_dir=args.hg_dir)
