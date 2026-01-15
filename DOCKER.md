# Docker Deployment Guide

Complete guide for deploying the Binance Sector Shot Detector using Docker and Docker Compose.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [CLI Commands](#cli-commands)
- [Data Persistence](#data-persistence)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Development](#development)
- [Security](#security)
- [Performance](#performance)
- [Migration from Local](#migration-from-local)

---

## Quick Start

Get the detector running in Docker in 5 minutes:

```bash
# 1. Clone or navigate to project directory
cd BinanceAlertManager

# 2. Create configuration file
cp config.example.yaml config.yaml
# Edit config.yaml with your settings (symbols, thresholds, optional API keys)

# 3. Build and start the detector
docker-compose up -d

# 4. View logs to verify startup
docker-compose logs -f detector

# 5. Check status
docker-compose ps
```

Expected output: Container "binance-detector" with status "Up" and healthy.

On first run, the detector will automatically backfill 13 hours of historical data (takes 1-2 minutes).

---

## Prerequisites

### Required Software

- **Docker Engine** 20.10+ (with BuildKit support)
- **Docker Compose** 2.0+

Check versions:
```bash
docker --version
docker-compose --version
```

### System Requirements

- **Disk Space**: 1GB minimum (500MB for image, 500MB for data volumes)
- **Memory**: 256MB minimum, 512MB recommended
- **CPU**: 1 core minimum
- **Network**: Outbound internet access for WebSocket connections to Binance

### Configuration File

You must have `config.yaml` in the project root:

```bash
# Copy from example
cp config.example.yaml config.yaml

# Edit with your settings
# Required: Configure symbols to monitor
# Optional: Add Binance API key/secret for advanced features
# Optional: Add Telegram bot credentials for notifications
```

---

## Usage

### Starting the Detector

```bash
# Build and start in detached mode
docker-compose up -d

# Build and start with logs visible
docker-compose up

# Force rebuild (after code changes)
docker-compose up -d --build
```

### Viewing Logs

```bash
# Follow logs in real-time
docker-compose logs -f detector

# View last 100 lines
docker-compose logs --tail=100 detector

# View logs with timestamps
docker-compose logs -f -t detector
```

### Stopping the Detector

```bash
# Stop container (preserves data)
docker-compose stop

# Stop and remove container (preserves data volumes)
docker-compose down

# Stop and remove container AND volumes (deletes all data)
docker-compose down -v
```

### Restarting the Detector

```bash
# Restart service
docker-compose restart

# Restart and force config reload
docker-compose restart detector
```

### Checking Status

```bash
# Check service status
docker-compose ps

# Check health status
docker inspect binance-detector --format='{{.State.Health.Status}}'

# Expected output: "healthy" after 30-60 seconds
```

---

## CLI Commands

The detector supports multiple operational modes. Override the default command to use different modes.

### Run Mode (Default)

Standard operation - monitors markets in real-time and generates alerts.

```bash
# Using docker-compose (default)
docker-compose up -d

# Using docker run directly
docker run -d \
  -v detector-data:/app/data \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  binance-detector
```

### Backfill Mode

Manually fetch historical data from Binance REST API.

```bash
# Backfill last 13 hours (default)
docker-compose run --rm detector python -m detector backfill --hours 13 --config config.yaml

# Backfill last 24 hours
docker-compose run --rm detector python -m detector backfill --hours 24 --config config.yaml

# Skip automatic backfill on run
docker-compose run --rm detector python -m detector run --config config.yaml --skip-backfill
```

**Note**: First-time runs automatically backfill 13 hours of data. Use `--skip-backfill` to disable this behavior.

### Database Migration Mode

Initialize or upgrade database schema.

```bash
# Initialize database
docker-compose run --rm detector python -m detector db-migrate --config config.yaml
```

**Note**: This is automatically done on first run, but useful for schema upgrades or manual initialization.

### Report Mode

Generate JSON reports of detected events.

```bash
# Generate report for last 24 hours
docker-compose run --rm detector python -m detector report --since 24h --output /app/report.json --config config.yaml

# Generate report for last 7 days
docker-compose run --rm detector python -m detector report --since 7d --output /app/report.json --config config.yaml

# Copy report to host
docker cp binance-detector:/app/report.json ./report.json
```

---

## Data Persistence

### Volume Architecture

The detector uses Docker named volumes for data persistence:

```yaml
volumes:
  detector-data:/app/data  # SQLite database and runtime data
  ./config.yaml:/app/config.yaml:ro  # Configuration (read-only)
```

**Key Points**:
- Database file: `/app/data/market.db` inside container
- Config file: `/app/config.yaml` inside container (mounted from host)
- Data persists across container restarts and removals
- Only deleted with `docker-compose down -v`

### Viewing Volume Data

```bash
# List volumes
docker volume ls | grep detector

# Inspect volume details
docker volume inspect binance-alert-manager_detector-data

# Check volume size
docker system df -v | grep detector-data
```

### Backing Up Data

```bash
# Create backup directory
mkdir -p backups

# Backup database (container must be running)
docker exec binance-detector sqlite3 /app/data/market.db ".backup /app/data/backup.db"
docker cp binance-detector:/app/data/backup.db ./backups/market-$(date +%Y%m%d).db

# Alternative: Backup entire volume
docker run --rm \
  -v binance-alert-manager_detector-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/detector-data-$(date +%Y%m%d).tar.gz -C /data .
```

### Restoring from Backup

```bash
# Stop detector
docker-compose down

# Restore database file
docker run --rm \
  -v binance-alert-manager_detector-data:/data \
  -v $(pwd)/backups:/backup \
  alpine sh -c "cd /data && tar xzf /backup/detector-data-YYYYMMDD.tar.gz"

# Restart detector
docker-compose up -d
```

### Clearing All Data

```bash
# Stop and remove volumes (deletes all historical data)
docker-compose down -v

# Restart fresh
docker-compose up -d
```

---

## Configuration

### Configuration File Location

The `config.yaml` file must exist in the project root before starting the container.

```bash
# Host path: ./config.yaml
# Container path: /app/config.yaml (read-only mount)
```

### Modifying Configuration

```bash
# 1. Edit config.yaml on host
vim config.yaml

# 2. Restart detector to load new config
docker-compose restart detector

# 3. Verify new config loaded
docker-compose logs detector | grep "Configuration loaded"
```

### Environment Variables

Currently, all configuration is done via `config.yaml`. Future versions may support environment variable overrides.

### Telegram Configuration

If using Telegram notifications:

1. Edit `config.yaml`:
```yaml
alerts:
  telegram:
    enabled: true
    bot_token: "YOUR_BOT_TOKEN"
    chat_id: "YOUR_CHAT_ID"
```

2. Restart detector:
```bash
docker-compose restart detector
```

3. Verify in logs:
```bash
docker-compose logs detector | grep "Telegram"
```

---

## Architecture

### Multi-Stage Build

The Dockerfile uses a two-stage build for optimal image size and security:

```
┌─────────────────────────────────┐
│   Stage 1: Builder              │
│   - Base: python:3.12-slim      │
│   - Install Poetry 1.8.2        │
│   - Install dependencies        │
│   - Copy application code       │
│   Size: ~600MB (not shipped)    │
└─────────────────────────────────┘
              │
              │ Copy .venv + code
              ▼
┌─────────────────────────────────┐
│   Stage 2: Runtime              │
│   - Base: python:3.12-slim      │
│   - No Poetry installed         │
│   - Non-root user (appuser)     │
│   - Virtual environment only    │
│   Size: ~200-300MB (shipped)    │
└─────────────────────────────────┘
```

**Benefits**:
- Smaller final image (no Poetry, no build tools)
- Faster deployments
- Reduced attack surface

### Non-Root User Security

Container runs as non-root user for security:

```dockerfile
# Created in Dockerfile
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser

USER appuser
```

Verify:
```bash
docker exec binance-detector whoami
# Output: appuser

docker exec binance-detector id
# Output: uid=1000(appuser) gid=1000(appuser)
```

### Volume Mounts

```
Host                          Container
─────────────────────────────────────────────
./config.yaml         →       /app/config.yaml (read-only)
detector-data volume  ↔       /app/data (read-write)
```

### Network Architecture

```
Container (binance-detector)
    │
    │ Outbound WebSocket (wss://)
    ▼
Binance Futures API
    │
    │ Real-time market data
    ▼
Container processes data → Alerts
```

**Note**: No inbound ports exposed. All connections are outbound (WebSocket client mode).

---

## Troubleshooting

### Container Won't Start

**Symptom**: Container exits immediately after starting

**Solutions**:

1. Check config.yaml exists:
```bash
ls -l config.yaml
```

2. Verify config.yaml syntax:
```bash
docker-compose config
```

3. Check logs for error:
```bash
docker-compose logs detector
```

4. Common issues:
   - Missing `config.yaml` → Copy from `config.example.yaml`
   - Invalid YAML syntax → Validate with yamllint
   - Permission errors → Check file ownership

---

### Database Locked Errors

**Symptom**: "Database is locked" in logs

**Cause**: Multiple containers accessing same SQLite database

**Solutions**:

1. Ensure only one instance running:
```bash
docker ps | grep binance-detector
# Should show only ONE container
```

2. Stop duplicate instances:
```bash
docker stop $(docker ps -q --filter name=binance-detector)
docker-compose down
```

3. Restart single instance:
```bash
docker-compose up -d
```

**Note**: SQLite WAL mode is enabled in config.yaml to reduce locking, but multiple concurrent writers still cause issues.

---

### Permission Errors

**Symptom**: "Permission denied" writing to /app/data

**Cause**: Volume ownership mismatch with non-root user

**Solution**:

Named volumes (recommended setup) handle permissions automatically. If using bind mounts:

```bash
# Fix ownership (Linux/macOS)
sudo chown -R 1000:1000 ./data

# Or run container as different user
docker-compose run --user $(id -u):$(id -g) detector
```

---

### WebSocket Connection Fails

**Symptom**: "Failed to connect to Binance WebSocket" in logs

**Causes & Solutions**:

1. **Firewall blocking outbound connections**:
   - Allow outbound HTTPS (443) and WebSocket (443)
   - Test: `curl -I https://fapi.binance.com`

2. **Network issues**:
   - Check container can reach internet:
   ```bash
   docker exec binance-detector ping -c 3 8.8.8.8
   ```

3. **Binance API maintenance**:
   - Check status: https://www.binance.com/en/support/announcement
   - Wait and retry after maintenance window

---

### Out of Disk Space

**Symptom**: Container stops writing data, logs show I/O errors

**Cause**: Docker volume or host disk full

**Solutions**:

1. Check volume size:
```bash
docker system df -v | grep detector
```

2. Check database size:
```bash
docker exec binance-detector du -sh /app/data
```

3. Clean old data:
```bash
# Option 1: Clear all data and restart fresh
docker-compose down -v
docker-compose up -d

# Option 2: Manually prune old bars (advanced)
docker exec binance-detector sqlite3 /app/data/market.db \
  "DELETE FROM bars_1m WHERE ts_open < strftime('%s', 'now', '-7 days') * 1000;"
docker exec binance-detector sqlite3 /app/data/market.db "VACUUM;"
```

4. Increase volume size (if on cloud provider with volume limits)

**Expected Growth**: ~1-2MB per day of continuous operation

---

### Health Check Failing

**Symptom**: `docker-compose ps` shows "unhealthy" status

**Cause**: Database not accessible or corruption

**Solutions**:

1. Check health check logs:
```bash
docker inspect binance-detector --format='{{json .State.Health}}' | jq
```

2. Manually test database:
```bash
docker exec binance-detector sqlite3 /app/data/market.db ".tables"
# Should list: alerts, bars_1m, events, features, positions, sector_diffusion
```

3. If corrupted, restore from backup or reinitialize:
```bash
docker-compose down -v
docker-compose up -d
```

---

### Image Build Fails

**Symptom**: `docker build` or `docker-compose build` fails

**Causes & Solutions**:

1. **Poetry dependency resolution fails**:
   - Check `poetry.lock` is up to date
   - Try rebuilding lock file locally: `poetry lock --no-update`

2. **Network timeout during build**:
   - Retry build (BuildKit caches progress)
   - Increase Docker timeout: `export DOCKER_BUILDKIT_TIMEOUT=600`

3. **Disk space during build**:
   - Clean build cache: `docker builder prune`
   - Check disk space: `df -h`

---

## Advanced Usage

### Building Custom Images

```bash
# Build with custom tag
docker build -t my-detector:v1.0 .

# Build with build arguments (if needed)
docker build --build-arg POETRY_VERSION=1.8.3 -t my-detector:v1.0 .

# Push to registry
docker tag my-detector:v1.0 myregistry.com/detector:v1.0
docker push myregistry.com/detector:v1.0
```

### Running in Production

Production deployment checklist:

- [ ] Use specific image tag (not `latest`)
- [ ] Set resource limits (CPU, memory)
- [ ] Configure restart policy
- [ ] Set up log rotation
- [ ] Monitor health checks
- [ ] Set up automated backups
- [ ] Use secrets management (not plain-text API keys)

Example production docker-compose.yml:

```yaml
version: '3.8'

services:
  detector:
    image: binance-detector:1.0.0  # Specific version
    restart: always
    user: "1000:1000"

    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M

    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

    volumes:
      - detector-data:/app/data
      - ./config.yaml:/app/config.yaml:ro

volumes:
  detector-data:
    driver: local
```

### Resource Limits

Set CPU and memory limits to prevent resource exhaustion:

```bash
# Run with limits using docker run
docker run -d \
  --cpus="1.0" \
  --memory="512m" \
  -v detector-data:/app/data \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  binance-detector

# Or use docker-compose deploy section (see above)
```

### Log Rotation

Configure log rotation to prevent disk fill:

```yaml
# docker-compose.yml
services:
  detector:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"    # Max log file size
        max-file: "3"      # Keep 3 rotated files
```

View rotated logs:
```bash
docker-compose logs --tail=100 detector
```

---

## Development

### Running Tests in Container

```bash
# Run all tests
docker-compose run --rm detector poetry run pytest tests/ -v

# Run specific test file
docker-compose run --rm detector poetry run pytest tests/test_detector_rules.py -v

# Run with coverage
docker-compose run --rm detector poetry run pytest tests/ --cov=detector --cov-report=html
```

### Mounting Code for Development

Edit `docker-compose.yml` to mount code directory:

```yaml
services:
  detector:
    volumes:
      - detector-data:/app/data
      - ./config.yaml:/app/config.yaml:ro
      - ./detector:/app/detector:ro  # Mount code for live reload
```

**Note**: Requires container restart after code changes.

### Debugging Containerized App

```bash
# Get shell inside running container
docker exec -it binance-detector /bin/bash

# Run Python interpreter
docker exec -it binance-detector python

# Check installed packages
docker exec binance-detector pip list

# Check Poetry environment
docker exec binance-detector poetry env info
```

---

## Security

### Security Best Practices

This Docker setup follows production security standards:

- ✅ **Non-root user** (UID 1000, appuser)
- ✅ **Minimal base image** (python:3.12-slim, not full)
- ✅ **Read-only config mount** (`:ro` flag)
- ✅ **No secrets in image** (use volumes for config)
- ✅ **Multi-stage build** (no Poetry in runtime)
- ✅ **Health checks** (monitor service status)

### Verifying Security

```bash
# 1. Check non-root user
docker exec binance-detector whoami
# Expected: appuser (not root)

# 2. Verify Poetry not in runtime image
docker exec binance-detector which poetry
# Expected: empty or "not found"

# 3. Check config is read-only
docker exec binance-detector touch /app/config.yaml
# Expected: "Read-only file system" error

# 4. Verify image size is minimal
docker images binance-detector
# Expected: < 500MB (typically 200-300MB)
```

### Handling Secrets

**Never embed secrets in Docker images or docker-compose.yml**.

Current approach: Mount `config.yaml` from host (excluded from image).

Future enhancement options:
- Docker secrets (Swarm mode)
- Environment variables from `.env` file
- External secrets management (HashiCorp Vault, AWS Secrets Manager)

---

## Performance

### Expected Metrics

- **Image Size**: 200-300MB (multi-stage build with slim base)
- **Build Time**: 3-5 minutes (cold build), < 1 minute (cached)
- **Startup Time**: 10-15 seconds (container ready)
- **Memory Usage**: 100-200MB (typical), 512MB (recommended limit)
- **CPU Usage**: < 10% (idle), 20-40% (active trading hours)
- **Disk Growth**: ~1-2MB per day (database growth)

### Monitoring Performance

```bash
# Check resource usage
docker stats binance-detector

# Check database size
docker exec binance-detector du -sh /app/data

# Check container uptime
docker ps --filter name=binance-detector --format "{{.Status}}"
```

### Optimizing Performance

1. **Enable BuildKit** (faster builds):
```bash
export DOCKER_BUILDKIT=1
docker-compose build
```

2. **Use volume for faster I/O**:
   - Named volumes are faster than bind mounts
   - Already configured in docker-compose.yml

3. **Allocate sufficient memory**:
   - Minimum: 256MB
   - Recommended: 512MB

---

## Migration from Local

### Exporting Existing Database

If you have an existing local installation with data:

```bash
# 1. Stop local detector
# Press Ctrl+C or kill process

# 2. Export database
cp data/market.db backups/market-backup-$(date +%Y%m%d).db

# 3. Verify backup
sqlite3 backups/market-backup-YYYYMMDD.db ".tables"
```

### Importing to Docker

```bash
# 1. Create Docker volume
docker volume create binance-alert-manager_detector-data

# 2. Copy database to volume
docker run --rm \
  -v binance-alert-manager_detector-data:/data \
  -v $(pwd)/backups:/backup \
  alpine cp /backup/market-backup-YYYYMMDD.db /data/market.db

# 3. Copy config
cp config.yaml config.yaml

# 4. Start detector
docker-compose up -d

# 5. Verify data imported
docker exec binance-detector sqlite3 /app/data/market.db \
  "SELECT COUNT(*) FROM bars_1m;"
```

### Running Alongside Local

To test Docker without affecting local installation:

```bash
# Use different config file
cp config.yaml config-docker.yaml

# Mount different config in docker-compose.yml
volumes:
  - ./config-docker.yaml:/app/config.yaml:ro

# Use different database by changing config-docker.yaml
storage:
  sqlite_path: "./data/market-docker.db"
```

---

## Additional Resources

- **Project README**: [README.md](README.md)
- **Configuration Guide**: [config.example.yaml](config.example.yaml)
- **Development Setup**: [CLAUDE.md](CLAUDE.md)
- **Docker Documentation**: https://docs.docker.com
- **Docker Compose Reference**: https://docs.docker.com/compose/compose-file/

---

## Support

For issues specific to Docker deployment:
1. Check this DOCKER.md troubleshooting section
2. Verify Docker and Docker Compose versions
3. Check Docker logs: `docker-compose logs detector`
4. Review container status: `docker-compose ps`

For application issues (not Docker-specific):
1. See main [README.md](README.md)
2. Check [CLAUDE.md](CLAUDE.md) for development guidance

---

**Docker deployment guide complete. Happy trading!**
