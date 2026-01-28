"""
Example script demonstrating how to use the image embeddings module.

This script shows:
1. Creating an Azure AI Search index
2. Processing a video to extract frames and generate embeddings
3. Searching for similar images
"""

import os
from azure.core.credentials import AzureKeyCredential
from image_embeddings import (
    AzureSearchIndexManager,
    ImageEmbeddingPipeline,
    load_config_from_env,
)


def main():
    """Main example function."""
    
    # Load configuration from environment variables
    print("Loading configuration...")
    config = load_config_from_env()
    
    # Create credentials
    search_credential = AzureKeyCredential(config["search_key"])
    inference_credential = AzureKeyCredential(config["inference_key"])
    
    # Step 1: Create Azure AI Search Index
    print("\n=== Step 1: Creating Azure AI Search Index ===")
    index_manager = AzureSearchIndexManager(
        endpoint=config["search_endpoint"],
        credential=search_credential,
    )
    
    # Create the index (or update if it exists)
    index_manager.create_index(
        index_name=config["index_name"],
        vector_dimensions=1024,  # Adjust based on your model
    )
    
    # List all indexes
    print("\nAvailable indexes:")
    for idx_name in index_manager.list_indexes():
        print(f"  - {idx_name}")
    
    # Step 2: Process Video and Generate Embeddings
    print("\n=== Step 2: Processing Video ===")
    
    # Initialize the pipeline
    pipeline = ImageEmbeddingPipeline(
        search_endpoint=config["search_endpoint"],
        search_credential=search_credential,
        inference_endpoint=config["inference_endpoint"],
        inference_credential=inference_credential,
        index_name=config["index_name"],
    )
    
    # Example: Process a video file
    video_path = "path/to/your/video.mp4"  # Replace with your video path
    
    # Check if video exists
    if os.path.exists(video_path):
        # Process the video
        # - Extract every 30th frame
        # - Save frames to 'frames' directory
        # - Limit to 50 frames for this example
        document_ids = pipeline.process_video(
            video_path=video_path,
            sample_rate=30,  # Extract every 30th frame
            max_frames=50,   # Limit to 50 frames
            output_dir="frames",  # Save frames here
        )
        
        print(f"\nProcessed {len(document_ids)} frames from video.")
    else:
        print(f"Video file not found: {video_path}")
        print("Please update the video_path variable with a valid video file.")
    
    # Step 3: Search for Similar Images (optional)
    print("\n=== Step 3: Searching for Similar Images ===")
    print("To search for similar images, provide a query image:")
    print("""
    import cv2
    query_image = cv2.imread('path/to/query/image.jpg')
    results = pipeline.search_similar_images(query_image, top_k=5)
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Video: {result['video_source']}")
        print(f"  Frame: {result['frame_number']}")
        print(f"  Timestamp: {result['timestamp']:.2f}s")
    """)
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
