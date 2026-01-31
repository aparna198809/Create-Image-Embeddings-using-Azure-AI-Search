# Create Image Embeddings using Azure AI Search and cohere embed model 

This repository provides reusable Python code for creating image embeddings from videos and storing them in Azure AI Search. It enables semantic search over video frames using Azure's AI services.

## Features

- **Azure AI Search Index Management**: Create and manage search indexes with vector search capabilities
- **Video Frame Extraction**: Extract frames from video files with configurable sampling rates
- **Image Embedding Generation**: Generate embeddings using Azure Foundry models (Azure AI Inference)
- **Semantic Search**: Search for similar images using vector similarity
- **Batch Processing**: Efficient batch processing of multiple images

## Architecture

The system consists of four main components:

1. **AzureSearchIndexManager**: Manages Azure AI Search index creation and configuration
2. **VideoFrameExtractor**: Extracts frames from video files using OpenCV
3. **ImageEmbeddingGenerator**: Generates embeddings using Azure AI Inference
4. **ImageEmbeddingPipeline**: Orchestrates the complete workflow

## Prerequisites

- Python 3.8 or higher
- Azure subscription with:
  - Azure AI Search service
  - Azure AI Foundry/Inference endpoint
- API keys or Azure AD credentials for both services

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aparna198809/Create-Image-Embeddings-using-Azure-AI-Search.git
cd Create-Image-Embeddings-using-Azure-AI-Search
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.template .env
# Edit .env with your Azure credentials
```

## Configuration

Create a `.env` file with your Azure credentials:

```env
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your_search_key
AZURE_SEARCH_INDEX_NAME=image-embeddings-index

AZURE_INFERENCE_ENDPOINT=https://your-inference-endpoint.inference.ai.azure.com
AZURE_INFERENCE_KEY=your_inference_key
```

## Usage

### Basic Example

```python
from azure.core.credentials import AzureKeyCredential
from image_embeddings import ImageEmbeddingPipeline, load_config_from_env

# Load configuration
config = load_config_from_env()

# Create credentials
search_credential = AzureKeyCredential(config["search_key"])
inference_credential = AzureKeyCredential(config["inference_key"])

# Initialize pipeline
pipeline = ImageEmbeddingPipeline(
    search_endpoint=config["search_endpoint"],
    search_credential=search_credential,
    inference_endpoint=config["inference_endpoint"],
    inference_credential=inference_credential,
    index_name=config["index_name"],
)

# Process a video
document_ids = pipeline.process_video(
    video_path="path/to/video.mp4",
    sample_rate=30,      # Extract every 30th frame
    max_frames=100,      # Maximum 100 frames
    output_dir="frames"  # Save frames here
)
```

### Create an Index

```python
from azure.core.credentials import AzureKeyCredential
from image_embeddings import AzureSearchIndexManager

index_manager = AzureSearchIndexManager(
    endpoint="https://your-search-service.search.windows.net",
    credential=AzureKeyCredential("your_key")
)

index_manager.create_index(
    index_name="image-embeddings-index",
    vector_dimensions=1024
)
```

### Extract Video Frames

```python
from image_embeddings import VideoFrameExtractor

extractor = VideoFrameExtractor("path/to/video.mp4")

# Extract every 30th frame
frames = extractor.extract_frames(
    sample_rate=30,
    output_dir="frames"
)

# Or extract frames at specific intervals
frames = extractor.extract_frames_by_interval(
    interval_seconds=1.0,  # One frame per second
    output_dir="frames"
)
```

### Generate Embeddings

```python
import cv2
from azure.core.credentials import AzureKeyCredential
from image_embeddings import ImageEmbeddingGenerator

generator = ImageEmbeddingGenerator(
    endpoint="https://your-inference-endpoint.inference.ai.azure.com",
    credential=AzureKeyCredential("your_key")
)

### Search Similar Images

```python
import cv2

# Search for similar images
input_query = read(user_query)
embedding = generator.generate_embedding(input_query)

results = pipeline.search_similar_images(input_query, top_k=5)

for result in results:
    print(f"Video: {result['file_name']}")
    print(f"Score: {result['@search.score']}")
```

## Running the Example

Run the example script:

```bash
python example_usage.py
```

Make sure to update the `video_path` variable in the script with the path to your video file.

## Project Structure

```
.
├── image_embeddings.py    # Main module with all functionality
├── example_usage.py       # Example usage script
├── requirements.txt       # Python dependencies
├── .env.template         # Environment variables template
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## Dependencies

- `azure-ai-inference`: Azure AI Inference SDK for embedding generation
- `azure-identity`: Azure authentication
- `azure-search-documents`: Azure AI Search SDK
- `opencv-python`: Video processing and frame extraction
- `Pillow`: Image processing
- `numpy`: Numerical operations
- `python-dotenv`: Environment variable management

## API Reference

### AzureSearchIndexManager

- `create_index(index_name, vector_dimensions)`: Create a new search index
- `delete_index(index_name)`: Delete an existing index
- `list_indexes()`: List all indexes

### VideoFrameExtractor

- `extract_frames(sample_rate, max_frames, output_dir)`: Extract frames with sampling
- `extract_frames_by_interval(interval_seconds, output_dir)`: Extract frames at time intervals

### ImageEmbeddingGenerator

- `generate_embedding(image)`: Generate embedding for a single image
- `generate_embeddings_batch(images, batch_size)`: Generate embeddings in batches

### ImageEmbeddingPipeline

- `process_video(video_path, sample_rate, max_frames, output_dir)`: Complete video processing workflow
- `search_similar_images(query_image, top_k)`: Search for similar images

## Troubleshooting

### Common Issues

1. **Video file not opening**: Ensure OpenCV is properly installed and the video codec is supported
2. **Authentication errors**: Verify your Azure credentials and endpoint URLs
3. **Memory issues**: Reduce `batch_size` or `max_frames` for large videos

### Error Messages

- "Unable to open video file": Check video path and file permissions
- "Missing required environment variables": Ensure all required variables are set in `.env`
- "Invalid video FPS": Video file may be corrupted or unsupported format

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is provided as-is for educational and development purposes.

## Acknowledgments

- Azure AI Search documentation
- Azure AI Inference documentation
- OpenCV community
