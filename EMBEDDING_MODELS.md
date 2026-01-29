# Embedding Models for Banking & Finance

## Recommended Models

Based on research and benchmarks for banking/finance semantic search, here are the recommended models:

### 🎯 Finance-Specific: `mukaj/fin-mpnet-base` (Best for Finance!)
- **Source**: [https://huggingface.co/mukaj/fin-mpnet-base](https://huggingface.co/mukaj/fin-mpnet-base)
- **Dimension**: 768
- **Speed**: Fast
- **Quality**: **State-of-the-art for financial documents** (79.91 on FiQA benchmark vs 49.96 for general models)
- **Use Case**: **SPECIFICALLY TRAINED** on 150k+ financial document QA examples (fine-tuned from all-mpnet-base-v2)
- **Performance**: Significantly outperforms general-purpose models on financial retrieval
- **Install**: When the Finance MPNet dropdown option is selected, the model is downloaded from the Hugging Face repo above. When the backend model registry is enabled (Redis), the model is stored in Redis after first download for fast subsequent loads.
- **Recommendation**: ⭐⭐⭐⭐⭐ **Use this for banking/finance applications!**

### 🏆 Best Overall General: `BAAI/bge-base-en-v1.5` (Default)
- **Dimension**: 768
- **Speed**: Fast
- **Quality**: Excellent for semantic understanding
- **Use Case**: Best balance of quality and speed for banking/finance queries
- **Install**: Automatically downloaded via sentence-transformers

### 🥇 Highest Quality: `BAAI/bge-large-en-v1.5`
- **Dimension**: 1024
- **Speed**: Slower (larger model)
- **Quality**: Highest quality semantic understanding
- **Use Case**: When quality is more important than speed
- **Trade-off**: ~2x slower but better semantic understanding

### 🎯 Excellent Alternative: `intfloat/e5-base-v2`
- **Dimension**: 768
- **Speed**: Fast
- **Quality**: Excellent for semantic understanding
- **Use Case**: Strong alternative to bge-base
- **Note**: Part of the E5 family, optimized for semantic tasks

### 📊 FinBERT: `ProsusAI/finbert`
- **Source**: [https://huggingface.co/ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- **Description**: Pre-trained NLP model for financial sentiment (positive/negative/neutral). BERT fine-tuned on Financial PhraseBank (Malo et al., 2014).
- **Use Case**: Financial sentiment analysis; used as the FinBERT option in the Compare/Chat model dropdown. When selected, the model is downloaded from the ProsusAI/finbert Hugging Face repo.

### 📊 Strong Semantic: `sentence-transformers/all-mpnet-base-v2`
- **Dimension**: 768
- **Speed**: Fast
- **Quality**: Strong semantic understanding
- **Use Case**: Good general-purpose model with strong semantic capabilities

### ⚡ Fastest (Lower Quality): `all-MiniLM-L6-v2`
- **Dimension**: 384
- **Speed**: Fastest
- **Quality**: Lower semantic understanding
- **Use Case**: When speed is critical and quality can be sacrificed
- **Note**: Current default, but not recommended for banking/finance

## Configuration

### Environment Variable
Set in `.env` file:
```bash
# For finance-specific applications (RECOMMENDED):
EMBEDDING_MODEL=mukaj/fin-mpnet-base

# Or use best general model:
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

### Why BGE Models for Banking/Finance?

1. **Better Semantic Understanding**: BGE models are trained specifically for semantic search and retrieval tasks
2. **Financial Domain**: While not finance-specific, they handle domain-specific terminology better than MiniLM
3. **Analytics Queries**: Better at understanding analytical queries like "top 5 states by customer count"
4. **Relationship Understanding**: Better at understanding relationships between entities (customers, transactions, amounts, locations)

## Model Comparison

| Model | Dimension | Speed | Quality | Banking/Finance Fit | Notes |
|-------|-----------|-------|---------|---------------------|-------|
| `mukaj/fin-mpnet-base` | 768 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Finance-tuned!** 79.91 FiQA score |
| `BAAI/bge-base-en-v1.5` | 768 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Best general model |
| `BAAI/bge-large-en-v1.5` | 1024 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Highest quality general |
| `intfloat/e5-base-v2` | 768 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Excellent semantic understanding |
| `all-mpnet-base-v2` | 768 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Strong semantic capabilities |
| `all-MiniLM-L6-v2` | 384 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Fastest but lower quality |

## Migration Notes

### If You Have Existing Data

⚠️ **Important**: Changing embedding models requires re-indexing your data!

1. **Backup your data** before changing models
2. **Update the model** in `.env`:
   ```bash
   EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
   ```
3. **Re-index your data** - The new model will generate different embeddings
4. **Test queries** to verify improved semantic understanding

### Dimension Changes

- `all-MiniLM-L6-v2`: 384 dimensions
- `BAAI/bge-base-en-v1.5`: 768 dimensions (2x increase)
- `BAAI/bge-large-en-v1.5`: 1024 dimensions

**Note**: Milvus collections need to match the embedding dimension. You'll need to recreate collections when changing models.

## Installation

All models are automatically downloaded via sentence-transformers:

```bash
pip install sentence-transformers
```

The first time you use a model, it will be downloaded automatically (~400MB-1.5GB depending on model).

## Performance Tips

1. **Use bge-base-en-v1.5** for best balance (default)
2. **Use bge-large-en-v1.5** if you need highest quality and can accept slower queries
3. **Monitor query times** - larger models are slower but provide better results
4. **Consider caching** frequently used queries if using larger models
