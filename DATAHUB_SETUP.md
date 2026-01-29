# DataHub Setup Guide

This guide explains how to set up and run DataHub for openInt metadata integration.

## Quick Start

### 1. Start DataHub

```bash
# Start all DataHub services
docker-compose -f docker-compose.datahub.yml up -d

# Check status
docker-compose -f docker-compose.datahub.yml ps

# View logs
docker-compose -f docker-compose.datahub.yml logs -f
```

### 2. Verify DataHub is Running

```bash
# Check health
curl http://localhost:9002/health

# Should return: GOOD
```

### 3. Access DataHub UI

Open your browser and navigate to:
- **DataHub UI**: http://localhost:9002
- **GraphQL Playground**: http://localhost:9002/api/graphiql

### 4. Configure Authentication

1. **Enable Token Authentication**:
   - Go to: http://localhost:9002/settings/authentication
   - Enable "Token-based Authentication"
   - Generate a personal access token

2. **Set Token for Ingestion**:
   ```bash
   export DATAHUB_TOKEN="your-token-here"
   ```

### 5. Ingest Metadata

```bash
cd openint-datahub
python ingest_metadata.py
```

Or use the DataHub CLI:
```bash
cd openint-datahub
datahub ingest -c ingestion_config.yaml
```

## Services Included

The `docker-compose.datahub.yml` includes:

- **Zookeeper**: Coordination service for Kafka
- **Kafka**: Message broker for DataHub events
- **Schema Registry**: Kafka schema management
- **Elasticsearch**: Search backend
- **MySQL**: Metadata storage database
- **datahub-gms**: General Metadata Service (backend API)
- **datahub-frontend**: Web UI and frontend proxy
- **datahub-mae-consumer**: Metadata Audit Event consumer
- **datahub-mce-consumer**: Metadata Change Event consumer

## Configuration

### Environment Variables

Key configuration in `docker-compose.datahub.yml`:

- `METADATA_SERVICE_AUTH_ENABLED=true` - Set for both `datahub-gms` and `datahub-frontend`
- Database credentials: `root` / `datahub`
- Ports: `9002` (GMS/Frontend), `9092` (Kafka), `3306` (MySQL), `9200` (Elasticsearch)

### Ports

- **9002**: DataHub GMS and Frontend
- **9092**: Kafka broker
- **3306**: MySQL database
- **9200**: Elasticsearch
- **8081**: Schema Registry
- **2181**: Zookeeper

## Management Commands

### Start Services
```bash
docker-compose -f docker-compose.datahub.yml up -d
```

### Stop Services
```bash
docker-compose -f docker-compose.datahub.yml down
```

### Stop and Remove Volumes (Clean Reset)
```bash
docker-compose -f docker-compose.datahub.yml down -v
```

### View Logs
```bash
# All services
docker-compose -f docker-compose.datahub.yml logs -f

# Specific service
docker-compose -f docker-compose.datahub.yml logs -f datahub-gms
docker-compose -f docker-compose.datahub.yml logs -f datahub-frontend
```

### Restart a Service
```bash
docker-compose -f docker-compose.datahub.yml restart datahub-frontend
```

### Check Service Status
```bash
docker-compose -f docker-compose.datahub.yml ps
```

## Troubleshooting

### Services Not Starting

1. **Check logs**:
   ```bash
   docker-compose -f docker-compose.datahub.yml logs
   ```

2. **Check port conflicts**:
   ```bash
   # Check if ports are already in use
   lsof -i :9002
   lsof -i :9092
   lsof -i :3306
   ```

3. **Verify Docker resources**:
   - Ensure Docker has enough memory (recommended: 4GB+)
   - Check disk space

### Authentication Issues

If you see `401 Unauthorized` errors:

1. **Verify METADATA_SERVICE_AUTH_ENABLED is set**:
   ```bash
   docker exec datahub-frontend env | grep METADATA_SERVICE_AUTH_ENABLED
   docker exec datahub-gms env | grep METADATA_SERVICE_AUTH_ENABLED
   ```
   Both should show: `METADATA_SERVICE_AUTH_ENABLED=true`

2. **Restart services after configuration changes**:
   ```bash
   docker-compose -f docker-compose.datahub.yml restart datahub-frontend datahub-gms
   ```

3. **Enable token authentication in UI**:
   - Navigate to http://localhost:9002/settings/authentication
   - Enable token authentication
   - Generate and use a token

### Health Check Failures

If health checks fail:

1. **Wait for services to fully start** (can take 2-3 minutes)
2. **Check individual service logs**
3. **Verify dependencies** (e.g., MySQL must be healthy before GMS starts)

## Integration with openInt

After DataHub is running:

1. **Ingest testdata metadata**:
   ```bash
   cd openint-datahub
   python ingest_metadata.py
   ```

2. **View datasets in DataHub**:
   - Browse: http://localhost:9002/dataset/openint
   - Search for platform: `openint`

3. **Explore schemas**:
   - Click on any dataset to see full schema
   - View field descriptions and types

## Data Persistence

By default, data is stored in Docker volumes. To persist data:

```bash
# View volumes
docker volume ls | grep datahub

# Backup volumes (optional)
docker run --rm -v datahub_mysql_data:/data -v $(pwd):/backup alpine tar czf /backup/mysql-backup.tar.gz /data
```

## Cleanup

To completely remove DataHub and all data:

```bash
# Stop and remove containers, networks, and volumes
docker-compose -f docker-compose.datahub.yml down -v

# Remove images (optional)
docker rmi acryldata/datahub-gms acryldata/datahub-frontend-react acryldata/datahub-mae-consumer acryldata/datahub-mce-consumer
```

## Next Steps

- See [openint-datahub/README.md](./openint-datahub/README.md) for ingestion details
- See [openint-datahub/DATAHUB_CONFIG.md](./openint-datahub/DATAHUB_CONFIG.md) for configuration options
- Check DataHub documentation: https://datahubproject.io/docs/
