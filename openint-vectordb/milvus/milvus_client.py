"""
Milvus Client Configuration
Connects to Milvus running on local Docker container
"""

import logging
import os
import re
import sys
import time
from dotenv import load_dotenv

logger = logging.getLogger("openint_vectordb.milvus")
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException
)
import numpy as np
from typing import List, Dict, Optional, Any

# Load environment variables from .env file
# override=True ensures .env file values take precedence over shell environment variables
load_dotenv(override=True)

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    SentenceTransformer = None
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Redis-backed model registry for O(1) boot when scaling (optional)
_load_model_from_registry = None
try:
    _backend = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "openint-backend"))
    if _backend not in sys.path:
        sys.path.insert(0, _backend)
    from model_registry import load_model_from_registry as _load_model_from_registry
except Exception:
    pass


class MilvusClient:
    """
    Milvus client configured to connect to local Docker instance.
    """
    
    def __init__(self, embedding_model: str = None):
        """
        Initialize Milvus client with local Docker configuration.
        Connects to Milvus running on Docker at http://localhost:19530
        
        Args:
            embedding_model: Name of the sentence-transformers model to use for embeddings.
                            Defaults to environment variable EMBEDDING_MODEL or "BAAI/bge-base-en-v1.5"
                            Recommended model for banking/finance:
                            - "mukaj/fin-mpnet-base" (default, finance-specific, best for banking/finance)
        """
        # Get model from environment or use finance model as default
        if embedding_model is None:
            embedding_model = os.getenv("EMBEDDING_MODEL", "mukaj/fin-mpnet-base")
        # Get connection parameters from environment or use defaults
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.alias = os.getenv("MILVUS_ALIAS", "default")
        
        # Get collection name from environment or use default
        # Milvus collection names can only contain numbers, letters, and underscores
        collection_name = os.getenv("MILVUS_COLLECTION", "default_collection")
        
        # CRITICAL: Sanitize collection name - Milvus will reject names with hyphens or special chars
        # Replace hyphens, dots, spaces, and other invalid characters with underscores
        sanitized_name = str(collection_name).replace("-", "_").replace(".", "_").replace(" ", "_")
        # Remove any other invalid characters (keep only alphanumeric and underscores)
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', sanitized_name)
        # Ensure it's not empty and starts with a letter or underscore (not a digit)
        if not sanitized_name or sanitized_name[0].isdigit():
            sanitized_name = "default_collection"
        
        # Always assign the sanitized name
        self.collection_name = sanitized_name
        
        # Warn if sanitization occurred
        if collection_name != self.collection_name:
            print(f"⚠️  Collection name sanitized: '{collection_name}' -> '{self.collection_name}'")
            print(f"   Milvus collection names can only contain letters, numbers, and underscores")
        
        # Final validation - ensure collection name is valid
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', self.collection_name):
            print(f"⚠️  WARNING: Collection name '{self.collection_name}' is still invalid, using 'default_collection'")
            self.collection_name = "default_collection"
        
        # Initialize embedding model
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        # Default dimensions for finance model
        self.embedding_dim = {
            "mukaj/fin-mpnet-base": 768,
        }.get(embedding_model, 768)  # Default to 768 for finance model
        
        if EMBEDDING_AVAILABLE and SentenceTransformer is not None:
            try:
                os.environ["TQDM_DISABLE"] = "1"
                logger.info("Loading embedding model: %s", embedding_model)
                if _load_model_from_registry is not None:
                    logger.info("Using Redis model registry (O(1) boot when scaling)")
                    self.embedding_model = _load_model_from_registry(embedding_model)
                else:
                    logger.info("Finance-specific model optimized for banking/finance queries")
                    self.embedding_model = SentenceTransformer(embedding_model)
                if self.embedding_model is not None:
                    self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                    logger.info("Embedding model loaded (dimension: %s). Model: %s", self.embedding_dim, embedding_model)
                else:
                    raise RuntimeError("Registry returned None")
            except Exception as e:
                logger.warning("Failed to load embedding model: %s. Install sentence-transformers. Trying fallback.", e)
                try:
                    self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                    self.embedding_dim = 384
                    self.embedding_model_name = "all-MiniLM-L6-v2"
                    logger.info("Fallback model loaded (dimension: %s)", self.embedding_dim)
                except Exception as e2:
                    logger.warning("Fallback model also failed: %s", e2)
        else:
            logger.warning("Embedding model not available. Install sentence-transformers for automatic embeddings.")
        
        # Initialize Milvus connection
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port
            )
            logger.info("Milvus client initialized. Host: %s:%s Collection: %s", self.host, self.port, self.collection_name)

        except Exception as e:
            error_msg = (
                "Failed to connect to Milvus at %s:%s. Error: %s. "
                "Ensure Milvus Docker container is running and .env has MILVUS_HOST/MILVUS_PORT."
            ) % (self.host, self.port, str(e))
            logger.warning("%s", error_msg)
            raise ConnectionError(error_msg) from e
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.embedding_model:
            raise ValueError(
                "Embedding model not available. Install sentence-transformers:\n"
                "   pip install sentence-transformers"
            )
        
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {e}") from e
    
    def _get_collection_schema(self):
        """
        Define the collection schema for Milvus.
        
        Returns:
            CollectionSchema object
        """
        # Define fields (customer_id and status for bank support filtering)
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="file_size", dtype=DataType.INT64),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="total_chunks", dtype=DataType.INT64),
            FieldSchema(name="original_id", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="customer_id", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=50),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Collection for document embeddings and metadata"
        )
        
        return schema
    
    def get_or_create_collection(self):
        """
        Get or create the Milvus collection.
        This is idempotent - safe to call multiple times.
        
        Returns:
            Collection object
        """
        # Validate collection name one more time before use (safety check)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', self.collection_name):
            # Re-sanitize if somehow invalid characters got through
            original_name = self.collection_name
            self.collection_name = re.sub(r'[^a-zA-Z0-9_]', '_', self.collection_name)
            if not self.collection_name or self.collection_name[0].isdigit():
                self.collection_name = "default_collection"
            print(f"⚠️  Collection name re-sanitized: '{original_name}' -> '{self.collection_name}'")
        
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name, using=self.alias):
                collection = Collection(self.collection_name, using=self.alias)
                print(f"✅ Using existing collection: {self.collection_name}")
                return collection
            else:
                # Create collection
                print(f"📝 Collection '{self.collection_name}' not found. Creating it...")
                schema = self._get_collection_schema()
                collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using=self.alias
                )
                
                # Create index on vector field
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                collection.create_index(
                    field_name="vector",
                    index_params=index_params
                )
                
                print(f"✅ Collection '{self.collection_name}' created successfully")
                return collection
                
        except MilvusException as e:
            error_msg = f"Error managing collection: {str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def get_collection(self):
        """
        Get the Milvus collection connection.
        Creates the collection if it doesn't exist.
        
        Returns:
            Collection connection object
        """
        return self.get_or_create_collection()
    
    def list_collections(self):
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        try:
            collections = utility.list_collections(using=self.alias)
            return collections
        except Exception as e:
            print(f"⚠️  Error listing collections: {e}")
            return []
    
    def _chunk_document(self, content: str, chunk_size: int = 60000, overlap: int = 200) -> List[str]:
        """
        Split a large document into chunks with optional overlap.
        
        Args:
            content: The document content to chunk
            chunk_size: Maximum size of each chunk (default: 60000, leaving room under 65535 limit)
            overlap: Number of characters to overlap between chunks (default: 200)
        
        Returns:
            List of chunk strings
        """
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # If not the last chunk, try to break at a sentence boundary
            if end < len(content):
                # Look for sentence endings within the last 500 chars
                search_start = max(start, end - 500)
                sentence_end = content.rfind('.', search_start, end)
                newline_end = content.rfind('\n', search_start, end)
                
                # Prefer sentence boundary, then newline, otherwise use exact position
                if sentence_end > search_start:
                    end = sentence_end + 1
                elif newline_end > search_start:
                    end = newline_end + 1
            
            chunk = content[start:end]
            chunks.append(chunk)
            
            # Move start position with overlap (except for last chunk)
            if end < len(content):
                start = end - overlap
            else:
                break
        
        return chunks
    
    def upsert_records(self, records: List[Dict[str, Any]], batch_size: int = 10000, auto_chunk: bool = True, chunk_size: int = 60000):
        """
        Insert or update records into the Milvus collection with automatic batching and optional chunking.
        
        Args:
            records: List of record dictionaries. Each record must have:
                - "id": Unique identifier for the record
                - "content" or "text": Text content for the record
                - Optional metadata fields (file_type, file_name, file_size, category)
            batch_size: Maximum number of records per batch (default: 10000; 10K–50K recommended for load performance)
            auto_chunk: If True, automatically chunk documents exceeding chunk_size (default: True)
            chunk_size: Maximum size of each chunk when auto_chunk is True (default: 60000)
        
        Returns:
            dict with keys:
                - "success": bool indicating overall success
                - "total_upserted": int total number of records upserted
                - "batches_processed": int number of batches processed
                - "errors": list of error messages if any
                - "chunked_documents": int number of documents that were chunked
        
        Raises:
            ValueError: If records are invalid
        """
        if not records:
            return {
                "success": True,
                "total_upserted": 0,
                "batches_processed": 0,
                "errors": [],
                "chunked_documents": 0
            }
        
        # Validate record structure
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Record at index {i} must be a dictionary")
            if "id" not in record:
                raise ValueError(f"Record at index {i} must have 'id' field")
            if "content" not in record and "text" not in record:
                raise ValueError(f"Record at index {i} must have 'content' or 'text' field")
        
        # Auto-chunk large documents if enabled
        MAX_CONTENT_LENGTH = 65535
        chunked_documents = 0
        expanded_records = []
        
        for record in records:
            content = record.get("content") or record.get("text", "")
            content = str(content) if content is not None else ""
            
            # Check if document needs chunking
            if auto_chunk and len(content) > chunk_size:
                chunks = self._chunk_document(content, chunk_size=chunk_size)
                chunked_documents += 1
                original_id = record["id"]
                
                print(f"📄 Chunking document '{original_id}': {len(content)} chars → {len(chunks)} chunks")
                
                # Create a record for each chunk
                for chunk_idx, chunk_content in enumerate(chunks):
                    chunk_record = record.copy()
                    chunk_record["id"] = f"{original_id}_chunk_{chunk_idx + 1}"
                    chunk_record["content"] = chunk_content
                    
                    # Add chunking metadata
                    metadata = chunk_record.get("metadata", {})
                    metadata["chunk_index"] = chunk_idx + 1
                    metadata["total_chunks"] = len(chunks)
                    metadata["original_id"] = original_id
                    metadata["is_chunk"] = True
                    chunk_record["metadata"] = metadata
                    
                    expanded_records.append(chunk_record)
            else:
                # Keep original record as-is
                expanded_records.append(record)
        
        # Insert expanded records (may include chunks)
        result = self.batch_upsert_records(expanded_records, batch_size)
        result["chunked_documents"] = chunked_documents
        
        return result
    
    def batch_upsert_records(self, records: List[Dict[str, Any]], batch_size: int = 10000):
        """
        Insert records in batches with error handling.
        
        Args:
            records: List of record dictionaries
            batch_size: Maximum number of records per batch (default: 10000; 10K–50K recommended for load performance)
        
        Returns:
            dict with success status, total upserted count, and any errors
        """
        collection = self.get_or_create_collection()
        total_upserted = 0
        batches_processed = 0
        errors = []
        
        # Process records in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                # Prepare batch data with validation
                ids = []
                contents = []
                texts_for_embedding = []
                file_types = []
                file_names = []
                file_sizes = []
                categories = []
                chunk_indices = []
                total_chunks_list = []
                original_ids = []
                customer_ids = []
                statuses = []
                
                for record_idx, record in enumerate(batch):
                    try:
                        # Validate and prepare ID
                        record_id = str(record["id"])
                        if len(record_id) > 500:
                            record_id = record_id[:500]
                            print(f"⚠️  Record ID truncated to 500 chars: {record['id'][:50]}...")
                        ids.append(record_id)
                        
                        # Content to store (full content; may include Structured Data JSON)
                        text = record.get("content") or record.get("text", "")
                        if text is None:
                            text = ""
                        text = str(text)
                        if not text or not text.strip():
                            file_name = record.get('metadata', {}).get('file_name', 'unknown')
                            text = f"[Empty content from file: {file_name}]"
                            print(f"⚠️  Empty content for record {record_id}, using placeholder")
                        MAX_CONTENT_LENGTH = 65535
                        if len(text) > MAX_CONTENT_LENGTH:
                            print(f"⚠️  Content truncated from {len(text)} to {MAX_CONTENT_LENGTH} chars for record {record_id}")
                            text = text[:MAX_CONTENT_LENGTH]
                        if len(text) > MAX_CONTENT_LENGTH:
                            text = text[:MAX_CONTENT_LENGTH]
                        contents.append(text)
                        # Embed natural-language text only when provided (better semantic match for bank support queries)
                        embed_text = record.get("text_for_embedding") or text
                        if embed_text is None:
                            embed_text = text
                        embed_text = str(embed_text)
                        if len(embed_text) > MAX_CONTENT_LENGTH:
                            embed_text = embed_text[:MAX_CONTENT_LENGTH]
                        texts_for_embedding.append(embed_text)
                        
                        # Extract and validate metadata
                        metadata = record.get("metadata", {})
                        
                        # File type: remove leading dot if present, limit to 50 chars
                        file_type = str(metadata.get("file_type", ""))
                        if file_type.startswith('.'):
                            file_type = file_type[1:]
                        if len(file_type) > 50:
                            file_type = file_type[:50]
                        file_types.append(file_type)
                        
                        # File name: limit to 500 chars
                        file_name = str(metadata.get("file_name", ""))
                        if len(file_name) > 500:
                            file_name = file_name[:500]
                            print(f"⚠️  File name truncated to 500 chars: {file_name[:50]}...")
                        file_names.append(file_name)
                        
                        # File size: ensure it's an integer
                        file_size = metadata.get("file_size", 0)
                        try:
                            file_size = int(file_size) if file_size is not None else 0
                        except (ValueError, TypeError):
                            file_size = 0
                            print(f"⚠️  Invalid file_size for record {record_id}, using 0")
                        file_sizes.append(file_size)
                        
                        # Category: limit to 100 chars
                        category = str(metadata.get("category", ""))
                        if len(category) > 100:
                            category = category[:100]
                        categories.append(category)
                        
                        # Chunk metadata (for chunked documents)
                        chunk_index = metadata.get("chunk_index", 0)
                        try:
                            chunk_index = int(chunk_index) if chunk_index is not None else 0
                        except (ValueError, TypeError):
                            chunk_index = 0
                        chunk_indices.append(chunk_index)
                        
                        total_chunks = metadata.get("total_chunks", 0)
                        try:
                            total_chunks = int(total_chunks) if total_chunks is not None else 0
                        except (ValueError, TypeError):
                            total_chunks = 0
                        total_chunks_list.append(total_chunks)
                        
                        # Original ID (for chunked documents)
                        original_id = str(metadata.get("original_id", ""))
                        if len(original_id) > 500:
                            original_id = original_id[:500]
                        original_ids.append(original_id)
                        # Bank support: filter by customer_id and status
                        customer_id = str(metadata.get("customer_id", ""))[:50]
                        customer_ids.append(customer_id)
                        status_val = str(metadata.get("status", ""))[:50]
                        statuses.append(status_val)
                        
                    except Exception as record_error:
                        error_msg = f"Error processing record {record.get('id', 'unknown')} in batch {batch_num}: {str(record_error)}"
                        errors.append(error_msg)
                        print(f"⚠️  {error_msg}")
                        # Skip this record
                        continue
                
                # Skip batch if no valid records
                if not ids:
                    print(f"⚠️  Batch {batch_num}: No valid records to insert")
                    continue
                
                # Final validation: ensure all contents are within length limit before insertion
                MAX_CONTENT_LENGTH = 65535
                for idx, content in enumerate(contents):
                    if len(content) > MAX_CONTENT_LENGTH:
                        print(f"⚠️  CRITICAL: Content at index {idx} (record {ids[idx]}) exceeds limit: {len(content)} > {MAX_CONTENT_LENGTH}. Truncating...")
                        contents[idx] = content[:MAX_CONTENT_LENGTH]
                        texts_for_embedding[idx] = content[:MAX_CONTENT_LENGTH]
                
                # Generate embeddings
                print(f"🔄 Generating embeddings for batch {batch_num} ({len(ids)} records)...")
                try:
                    vectors = self._generate_embeddings(texts_for_embedding)
                except Exception as embed_error:
                    error_msg = f"Batch {batch_num} embedding generation failed: {str(embed_error)}"
                    errors.append(error_msg)
                    print(f"⚠️  {error_msg}")
                    continue
                
                # Prepare data for insertion (order must match schema)
                # Backward compatibility: existing collections may not have customer_id/status fields
                schema_field_names = [f.name for f in collection.schema.fields]
                data = [
                    ids,
                    contents,
                    vectors,
                    file_types,
                    file_names,
                    file_sizes,
                    categories,
                    chunk_indices,
                    total_chunks_list,
                    original_ids,
                ]
                if "customer_id" in schema_field_names and "status" in schema_field_names:
                    data.extend([customer_ids, statuses])
                
                collection.insert(data)
                collection.flush()  # Ensure data is written
                
                total_upserted += len(ids)
                batches_processed += 1
                print(f"✅ Batch {batch_num}: Upserted {len(ids)} records")
                    
            except Exception as e:
                error_msg = f"Batch {batch_num} failed: {str(e)}"
                errors.append(error_msg)
                print(f"⚠️  {error_msg}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
        
        success = len(errors) == 0
        
        return {
            "success": success,
            "total_upserted": total_upserted,
            "batches_processed": batches_processed,
            "errors": errors
        }
    
    def list_records(self, limit: Optional[int] = None):
        """
        List all records in the collection.
        
        Args:
            limit: Maximum number of records to return (None for all)
        
        Returns:
            List of records with their IDs, content, and metadata
        """
        collection = self.get_or_create_collection()
        collection.load()  # Load collection into memory
        
        try:
            # Query all records
            results = collection.query(
                expr="id != ''",  # Get all records
                output_fields=["id", "content", "file_type", "file_name", "file_size", "category", 
                              "chunk_index", "total_chunks", "original_id"],
                limit=limit if limit else 16384  # Milvus default limit
            )
            
            records = []
            for result in results:
                record = {
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "metadata": {
                        "file_type": result.get("file_type", ""),
                        "file_name": result.get("file_name", ""),
                        "file_size": result.get("file_size", 0),
                        "category": result.get("category", ""),
                        "chunk_index": result.get("chunk_index", 0),
                        "total_chunks": result.get("total_chunks", 0),
                        "original_id": result.get("original_id", ""),
                        "is_chunk": result.get("chunk_index", 0) > 0
                    }
                }
                records.append(record)
            
            return records
        except Exception as e:
            print(f"⚠️  Error listing records: {e}")
            return []
    
    def delete_all_records(self):
        """
        Delete all records from the collection.
        
        Returns:
            dict with keys:
                - "success": bool indicating overall success
                - "deleted_count": int number of records deleted
                - "error": str error message if any
        """
        collection = self.get_or_create_collection()
        collection.load()
        
        try:
            # Get all record IDs
            results = collection.query(
                expr="id != ''",
                output_fields=["id"],
                limit=16384
            )
            all_ids = [result.get("id") for result in results]
            
            if not all_ids:
                print("ℹ️  No records found in collection. Nothing to delete.")
                return {
                    "success": True,
                    "deleted_count": 0,
                    "error": None
                }
            
            # Delete all records by their IDs
            # Milvus delete expression: delete in batches if too many
            deleted_count = 0
            batch_size = 1000  # Milvus can handle large batches, but be safe
            
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i + batch_size]
                # Format: id in ["id1", "id2", ...]
                id_list = ', '.join([f'"{id_val}"' for id_val in batch_ids])
                expr = f'id in [{id_list}]'
                collection.delete(expr=expr)
                deleted_count += len(batch_ids)
            
            collection.flush()
            
            print(f"✅ Successfully deleted {deleted_count} record(s) from collection '{self.collection_name}'")
            
            return {
                "success": True,
                "deleted_count": deleted_count,
                "error": None
            }
        except Exception as e:
            error_msg = f"Error deleting records: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "deleted_count": 0,
                "error": error_msg
            }
    
    def search(self, query_text: str, top_k: int = 5, limit: Optional[int] = None, expr: Optional[str] = None):
        """
        Search for similar records using text query.

        Args:
            query_text: Text query to search for
            top_k: Number of results to return
            limit: Optional limit for search results (deprecated, use top_k)
            expr: Optional boolean expression to filter (e.g. 'file_type == "zip_codes"')

        Returns:
            Tuple of (search_results, total_time_ms, embedding_time_ms, vector_search_time_ms)
        """
        collection = self.get_or_create_collection()
        collection.load()

        try:
            # Time embedding and search separately so callers can see where time is spent
            t_embed = time.perf_counter()
            query_vector = self._generate_embeddings([query_text])[0]
            embedding_time_ms = max(1, round((time.perf_counter() - t_embed) * 1000))
            # Search
            t_search = time.perf_counter()
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            search_kw = dict(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "content", "file_type", "file_name", "category",
                              "chunk_index", "total_chunks", "original_id"]
            )
            if expr:
                search_kw["expr"] = expr
            results = collection.search(**search_kw)
            vector_search_time_ms = max(1, round((time.perf_counter() - t_search) * 1000))
            # Total (embed + search) for backward compatibility
            query_time_ms = embedding_time_ms + vector_search_time_ms

            # Format results
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "id": hit.entity.get("id"),
                        "content": hit.entity.get("content", ""),
                        "score": hit.score,
                        "metadata": {
                            "file_type": hit.entity.get("file_type", ""),
                            "file_name": hit.entity.get("file_name", ""),
                            "category": hit.entity.get("category", ""),
                            "chunk_index": hit.entity.get("chunk_index", 0),
                            "total_chunks": hit.entity.get("total_chunks", 0),
                            "original_id": hit.entity.get("original_id", ""),
                            "is_chunk": hit.entity.get("chunk_index", 0) > 0
                        }
                    })
            return search_results, query_time_ms, embedding_time_ms, vector_search_time_ms
        except Exception as e:
            print(f"⚠️  Error searching: {e}")
            return [], 0, 0, 0
    
    def get_full_document(self, original_id: str) -> Optional[Dict[str, Any]]:
        """
        Reconstruct a full document from its chunks.
        
        Args:
            original_id: The original document ID (before chunking)
        
        Returns:
            Dictionary with:
                - "id": Original document ID
                - "content": Reconstructed full content
                - "metadata": Combined metadata from all chunks
                - "chunks_found": Number of chunks found
            Returns None if document not found
        """
        collection = self.get_or_create_collection()
        collection.load()
        
        try:
            # Query all chunks for this document
            results = collection.query(
                expr=f'original_id == "{original_id}"',
                output_fields=["id", "content", "file_type", "file_name", "file_size", "category",
                              "chunk_index", "total_chunks", "original_id"],
                limit=16384
            )
            
            if not results:
                # Try querying by ID directly (in case it wasn't chunked)
                results = collection.query(
                    expr=f'id == "{original_id}"',
                    output_fields=["id", "content", "file_type", "file_name", "file_size", "category",
                                  "chunk_index", "total_chunks", "original_id"],
                    limit=1
                )
                
                if results:
                    result = results[0]
                    return {
                        "id": result.get("id", ""),
                        "content": result.get("content", ""),
                        "metadata": {
                            "file_type": result.get("file_type", ""),
                            "file_name": result.get("file_name", ""),
                            "file_size": result.get("file_size", 0),
                            "category": result.get("category", ""),
                            "is_chunk": False
                        },
                        "chunks_found": 1
                    }
                return None
            
            # Sort chunks by chunk_index
            chunks = sorted(results, key=lambda x: x.get("chunk_index", 0))
            
            # Reconstruct full content
            full_content = "".join([chunk.get("content", "") for chunk in chunks])
            
            # Get metadata from first chunk (they should all be the same)
            first_chunk = chunks[0] if chunks else {}
            
            return {
                "id": original_id,
                "content": full_content,
                "metadata": {
                    "file_type": first_chunk.get("file_type", ""),
                    "file_name": first_chunk.get("file_name", ""),
                    "file_size": first_chunk.get("file_size", 0),
                    "category": first_chunk.get("category", ""),
                    "is_chunk": False,
                    "was_chunked": True,
                    "total_chunks": len(chunks)
                },
                "chunks_found": len(chunks)
            }
            
        except Exception as e:
            print(f"⚠️  Error reconstructing document: {e}")
            return None


# Convenience function for quick access
def get_milvus_client(embedding_model: str = None):
    """
    Get a configured Milvus client instance.
    
    Args:
        embedding_model: Name of the sentence-transformers model to use.
                        Defaults to environment variable EMBEDDING_MODEL or "BAAI/bge-base-en-v1.5"
                        Recommended for banking/finance:
                        - "BAAI/bge-base-en-v1.5" (default, best balance)
                        - "BAAI/bge-large-en-v1.5" (higher quality)
                        - "intfloat/e5-base-v2" (excellent semantic understanding)
    
    Returns:
        MilvusClient instance
    """
    return MilvusClient(embedding_model=embedding_model)


# Also export as module-level for backward compatibility
__all__ = ["MilvusClient", "get_milvus_client"]


# Example usage
if __name__ == "__main__":
    # Initialize client (automatically connects to local Docker instance)
    client = MilvusClient()
    
    # List available collections
    collections = client.list_collections()
    print(f"\n📋 Available collections: {collections}")
    
    # Get collection connection
    try:
        collection = client.get_collection()
        print(f"\n✅ Successfully connected to collection: {client.collection_name}")
    except Exception as e:
        print(f"\n⚠️  Error connecting to collection: {e}")
        print("   Make sure Milvus Docker container is running on port 19530")
