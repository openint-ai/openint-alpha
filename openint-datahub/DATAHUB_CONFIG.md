# DataHub Configuration Guide

## Environment Variables

To enable proper communication between DataHub frontend and GMS (General Metadata Service), set the following environment variable:

### For datahub-frontend container:

```bash
METADATA_SERVICE_AUTH_ENABLED=true
```

## Docker Compose Configuration

If you're using Docker Compose to run DataHub, add this to your `docker-compose.yml`:

```yaml
services:
  datahub-frontend:
    environment:
      - METADATA_SERVICE_AUTH_ENABLED=true
    # ... other configuration
```

Or set it via environment file (`.env`):

```bash
METADATA_SERVICE_AUTH_ENABLED=true
```

## Quick Setup

1. **If using Docker Compose:**
   ```bash
   # Add to docker-compose.yml under datahub-frontend service
   environment:
     - METADATA_SERVICE_AUTH_ENABLED=true
   
   # Restart the container
   docker-compose restart datahub-frontend
   ```

2. **If using Docker run:**
   ```bash
   docker run -e METADATA_SERVICE_AUTH_ENABLED=true \
     # ... other options
     datahub-frontend
   ```

3. **If using environment file:**
   ```bash
   echo "METADATA_SERVICE_AUTH_ENABLED=true" >> .env
   docker-compose up -d
   ```

## Verification

After setting the environment variable, verify the configuration:

```bash
# Check if the variable is set in the container
docker exec datahub-frontend env | grep METADATA_SERVICE_AUTH_ENABLED

# Should output:
# METADATA_SERVICE_AUTH_ENABLED=true
```

## Authentication Setup

Once `METADATA_SERVICE_AUTH_ENABLED=true` is set, you can:

1. **Enable token authentication** in DataHub UI (Settings → Authentication)
2. **Generate a personal access token**
3. **Set it as environment variable:**
   ```bash
   export DATAHUB_TOKEN="your-token-here"
   ```
4. **Run the ingestion script:**
   ```bash
   cd openint-datahub
   python ingest_metadata.py
   ```

## Alternative: Use DataHub CLI

If authentication is still causing issues, use the DataHub CLI which handles authentication automatically:

```bash
cd openint-datahub
datahub ingest -c ingestion_config.yaml
```
