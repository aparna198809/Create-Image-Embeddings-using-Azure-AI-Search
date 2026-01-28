"""
Azure AI Search Image Embeddings Module

This module provides functionality for:
1. Creating and managing Azure AI Search indexes
2. Extracting frames from video files
3. Generating image embeddings using Azure Foundry models
4. Storing embeddings in Azure AI Search
"""

import os
import json
import base64
from typing import List, Dict, Optional, Tuple
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from azure.identity import DefaultAzureCredential, AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.search.documents.models import VectorizedQuery
from azure.ai.inference import EmbeddingsClient


class AzureSearchIndexManager:
    """Manages Azure AI Search index creation and operations."""
    
    def __init__(self, endpoint: str, credential):
        """
        Initialize the Azure Search Index Manager.
        
        Args:
            endpoint: Azure Search service endpoint
            credential: Azure credential (AzureKeyCredential or DefaultAzureCredential)
        """
        self.endpoint = endpoint
        self.credential = credential
        self.index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    
    def create_index(self, index_name: str, vector_dimensions: int = 1024) -> SearchIndex:
        """
        Create an Azure AI Search index for image embeddings.
        
        Args:
            index_name: Name of the index to create
            vector_dimensions: Dimensionality of the embedding vectors
            
        Returns:
            Created SearchIndex object
        """
        # Define fields for the index
        fields = [
            SearchField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                searchable=False,
                filterable=True,
            ),
            SearchField(
                name="image_path",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
            ),
            SearchField(
                name="frame_number",
                type=SearchFieldDataType.Int32,
                searchable=False,
                filterable=True,
                sortable=True,
            ),
            SearchField(
                name="timestamp",
                type=SearchFieldDataType.Double,
                searchable=False,
                filterable=True,
                sortable=True,
            ),
            SearchField(
                name="video_source",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_dimensions,
                vector_search_profile_name="myHnswProfile",
            ),
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine",
                    },
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                )
            ],
        )
        
        # Create the index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
        )
        
        result = self.index_client.create_or_update_index(index)
        print(f"Index '{index_name}' created successfully.")
        return result
    
    def delete_index(self, index_name: str):
        """Delete an index."""
        self.index_client.delete_index(index_name)
        print(f"Index '{index_name}' deleted successfully.")
    
    def list_indexes(self) -> List[str]:
        """List all indexes in the search service."""
        indexes = self.index_client.list_indexes()
        return [index.name for index in indexes]


class VideoFrameExtractor:
    """Extracts frames from video files."""
    
    def __init__(self, video_path: str):
        """
        Initialize the video frame extractor.
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        
        if not self.video.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
    
    def extract_frames(
        self,
        sample_rate: int = 1,
        max_frames: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> List[Tuple[np.ndarray, int, float]]:
        """
        Extract frames from the video.
        
        Args:
            sample_rate: Extract every Nth frame (default: 1 = every frame)
            max_frames: Maximum number of frames to extract
            output_dir: Directory to save extracted frames (optional)
            
        Returns:
            List of tuples (frame_array, frame_number, timestamp)
        """
        frames = []
        frame_count = 0
        extracted_count = 0
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                timestamp = frame_count / self.fps if self.fps > 0 else 0
                frames.append((frame, frame_count, timestamp))
                
                if output_dir:
                    output_path = os.path.join(
                        output_dir, f"frame_{frame_count:06d}.jpg"
                    )
                    cv2.imwrite(output_path, frame)
                
                extracted_count += 1
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        self.video.release()
        print(f"Extracted {len(frames)} frames from video.")
        return frames
    
    def extract_frames_by_interval(
        self,
        interval_seconds: float,
        output_dir: Optional[str] = None,
    ) -> List[Tuple[np.ndarray, int, float]]:
        """
        Extract frames at specified time intervals.
        
        Args:
            interval_seconds: Time interval between frames in seconds
            output_dir: Directory to save extracted frames (optional)
            
        Returns:
            List of tuples (frame_array, frame_number, timestamp)
        """
        if self.fps <= 0:
            raise ValueError("Invalid video FPS")
        
        sample_rate = int(interval_seconds * self.fps)
        return self.extract_frames(sample_rate=sample_rate, output_dir=output_dir)
    
    def __del__(self):
        """Release video resources."""
        if hasattr(self, 'video'):
            self.video.release()


class ImageEmbeddingGenerator:
    """Generates image embeddings using Azure AI Inference."""
    
    def __init__(self, endpoint: str, credential):
        """
        Initialize the image embedding generator.
        
        Args:
            endpoint: Azure AI Inference endpoint
            credential: Azure credential
        """
        self.endpoint = endpoint
        self.credential = credential
        self.client = EmbeddingsClient(endpoint=endpoint, credential=credential)
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """
        Convert OpenCV image (numpy array) to base64 string.
        
        Args:
            image: Image as numpy array (OpenCV format)
            
        Returns:
            Base64 encoded string
        """
        # Convert BGR (OpenCV) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def generate_embedding(self, image: np.ndarray) -> List[float]:
        """
        Generate embedding for a single image.
        
        Args:
            image: Image as numpy array (OpenCV format)
            
        Returns:
            Embedding vector as list of floats
        """
        # Convert image to base64
        image_base64 = self.image_to_base64(image)
        
        # Generate embedding
        response = self.client.embed(
            input=[{"image": image_base64}],
        )
        
        return response.data[0].embedding
    
    def generate_embeddings_batch(
        self, images: List[np.ndarray], batch_size: int = 10
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple images in batches.
        
        Args:
            images: List of images as numpy arrays
            batch_size: Number of images to process in each batch
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Convert images to base64
            image_inputs = [
                {"image": self.image_to_base64(img)} for img in batch
            ]
            
            # Generate embeddings
            response = self.client.embed(input=image_inputs)
            
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            
            print(f"Processed {min(i + batch_size, len(images))}/{len(images)} images")
        
        return embeddings


class ImageEmbeddingPipeline:
    """Complete pipeline for processing videos and storing embeddings."""
    
    def __init__(
        self,
        search_endpoint: str,
        search_credential,
        inference_endpoint: str,
        inference_credential,
        index_name: str,
    ):
        """
        Initialize the image embedding pipeline.
        
        Args:
            search_endpoint: Azure Search service endpoint
            search_credential: Azure Search credential
            inference_endpoint: Azure AI Inference endpoint
            inference_credential: Azure AI Inference credential
            index_name: Name of the search index
        """
        self.index_manager = AzureSearchIndexManager(
            search_endpoint, search_credential
        )
        self.embedding_generator = ImageEmbeddingGenerator(
            inference_endpoint, inference_credential
        )
        self.index_name = index_name
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=search_credential,
        )
    
    def process_video(
        self,
        video_path: str,
        sample_rate: int = 30,
        max_frames: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Process a video file: extract frames, generate embeddings, and store in index.
        
        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame
            max_frames: Maximum number of frames to process
            output_dir: Directory to save extracted frames (optional)
            
        Returns:
            List of document IDs uploaded to the index
        """
        print(f"Processing video: {video_path}")
        
        # Extract frames
        extractor = VideoFrameExtractor(video_path)
        frames_data = extractor.extract_frames(
            sample_rate=sample_rate,
            max_frames=max_frames,
            output_dir=output_dir,
        )
        
        if not frames_data:
            print("No frames extracted from video.")
            return []
        
        # Generate embeddings
        print("Generating embeddings...")
        frames = [frame for frame, _, _ in frames_data]
        embeddings = self.embedding_generator.generate_embeddings_batch(frames)
        
        # Prepare documents for upload
        documents = []
        video_name = os.path.basename(video_path)
        
        for (frame, frame_num, timestamp), embedding in zip(frames_data, embeddings):
            # Generate unique document ID
            doc_id = f"{video_name}_{frame_num}_{int(timestamp * 1000)}"
            
            # Determine image path
            if output_dir:
                image_full_path = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
            else:
                image_full_path = f"{video_path}#frame_{frame_num}"
            
            document = {
                "id": doc_id,
                "image_path": image_full_path,
                "frame_number": frame_num,
                "timestamp": timestamp,
                "video_source": video_name,
                "embedding": embedding,
            }
            documents.append(document)
        
        # Upload to index
        print(f"Uploading {len(documents)} documents to index...")
        result = self.search_client.upload_documents(documents)
        
        # Check for failures
        uploaded_ids = []
        failed_ids = []
        for doc, upload_result in zip(documents, result):
            if upload_result.succeeded:
                uploaded_ids.append(doc["id"])
            else:
                failed_ids.append((doc["id"], upload_result.error_message))
        
        if failed_ids:
            print(f"Warning: {len(failed_ids)} documents failed to upload:")
            for doc_id, error in failed_ids[:5]:  # Show first 5 errors
                print(f"  - {doc_id}: {error}")
        
        print(f"Successfully uploaded {len(uploaded_ids)} documents.")
        return uploaded_ids
    
    def search_similar_images(
        self,
        query_image: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Search for similar images using a query image.
        
        Args:
            query_image: Query image as numpy array
            top_k: Number of top results to return
            
        Returns:
            List of search results
        """
        # Generate embedding for query image
        query_embedding = self.embedding_generator.generate_embedding(query_image)
        
        # Create vector query
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="embedding"
        )
        
        # Perform vector search
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["id", "image_path", "frame_number", "timestamp", "video_source"],
        )
        
        return [result for result in results]


def load_config_from_env() -> Dict[str, str]:
    """
    Load configuration from environment variables.
    
    Returns:
        Dictionary with configuration values
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    config = {
        "search_endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "search_key": os.getenv("AZURE_SEARCH_KEY"),
        "index_name": os.getenv("AZURE_SEARCH_INDEX_NAME", "image-embeddings-index"),
        "inference_endpoint": os.getenv("AZURE_INFERENCE_ENDPOINT"),
        "inference_key": os.getenv("AZURE_INFERENCE_KEY"),
    }
    
    # Validate required fields
    required_fields = ["search_endpoint", "search_key", "inference_endpoint", "inference_key"]
    missing_fields = [field for field in required_fields if not config.get(field)]
    
    if missing_fields:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
    
    return config
