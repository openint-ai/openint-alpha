# Deploy OpenInt (merged backend+UI) to AWS EC2

This guide deploys the merged backend and UI to a single EC2 instance. The Flask app serves the built React UI from `/` and the API from `/api/*`.

## Prerequisites

- EC2 instance running (e.g. Amazon Linux 2 or Ubuntu) with Python 3.8+, and SSH access.
- Your SSH key: `ssh -i ~/.ssh/openint.pem ec2-user@<EC2_HOST>` works.

## 1. Build and deploy from your machine

```bash
# Build the UI (produces openint-ui/dist)
./build.sh

# Set EC2 host and key (required)
export OPENINT_EC2_KEY=~/.ssh/openint.pem
export OPENINT_EC2_HOST=ec2-user@ec2-3-148-183-18.us-east-2.compute.amazonaws.com

# Optional: custom remote directory (default: /home/ec2-user/openint)
# export OPENINT_EC2_REMOTE_DIR=/opt/openint

# Deploy: rsync code + built UI to EC2 and start the app
./deploy_to_ec2.sh
```

The deploy script will:

- Rsync `openint-backend`, `openint-agents`, `openint-vectordb`, `openint-graph`, `shared`, and `openint-ui/dist` to the EC2 instance.
- SSH into EC2 and run `scripts/ec2_install_and_restart.sh`, which installs Python dependencies and starts the app (gunicorn or python) on port 3001.

## 2. On EC2: environment variables

The app reads env vars from the shell or a `.env` file. Create `~/openint/.env` on EC2 (or set in systemd) with at least:

```bash
# Required for merged UI
SERVE_UI=1
FLASK_ENV=production
PORT=3001

# Backend / agents (adjust for your setup)
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
# MILVUS_HOST=...
# NEO4J_URI=...
# OLLAMA_HOST=...   # if using sa-agent sentence generation
```

If Redis/Milvus/Neo4j run on the same EC2, use `127.0.0.1` or `localhost`. If they run elsewhere, set the host/IP and ensure the EC2 security group allows outbound access to those ports.

## 3. Open the app

- **API health:** `http://<EC2_PUBLIC_IP>:3001/api/health`
- **UI (merged):** `http://<EC2_PUBLIC_IP>:3001/`

Ensure the EC2 security group allows inbound TCP on port 3001 (or whatever `PORT` you use).

## 4. Logs and restart

- **Logs:** `ssh -i ~/.ssh/openint.pem ec2-user@<EC2_HOST> tail -f ~/openint/openint.log`
- **Restart after deploy:** Each run of `./deploy_to_ec2.sh` stops the previous process (by port) and starts a new one. To restart manually on EC2:
  ```bash
  cd ~/openint && ./scripts/ec2_install_and_restart.sh
  ```
  Run in the background or under systemd (see below).

## 5. Optional: systemd service

To run OpenInt as a service that survives reboots and allows `systemctl restart openint`:

1. On EC2, create `/etc/systemd/system/openint.service`:

```ini
[Unit]
Description=OpenInt merged backend+UI
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/openint
Environment=SERVE_UI=1
Environment=FLASK_ENV=production
Environment=PORT=3001
Environment=PYTHONPATH=/home/ec2-user/openint/openint-agents:/home/ec2-user/openint/openint-vectordb/milvus:/home/ec2-user/openint/openint-graph
ExecStart=/home/ec2-user/openint/venv/bin/gunicorn -w 1 -b 0.0.0.0:3001 --chdir /home/ec2-user/openint/openint-backend main:app
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

2. Create venv and install deps once (if not already done):
   ```bash
   cd ~/openint && ./scripts/ec2_install_and_restart.sh
   # Stop the foreground process (Ctrl+C), then:
   sudo systemctl daemon-reload
   sudo systemctl enable openint
   sudo systemctl start openint
   ```

3. After future deploys, restart the service:
   ```bash
   sudo systemctl restart openint
   ```

## 6. Optional: Nginx reverse proxy

To serve on port 80/443 and add TLS, put Nginx in front:

- Proxy `http://127.0.0.1:3001` for both `/` and `/api`.
- Configure SSL with Let's Encrypt (certbot) if needed.

Example server block (HTTP only):

```nginx
server {
    listen 80;
    server_name your-domain.com;
    location / {
        proxy_pass http://127.0.0.1:3001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
    }
}
```

Then open `http://your-domain.com` (or https with certbot).
