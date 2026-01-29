#!/usr/bin/env python3
"""
Generate self-signed TLS certificates for local HTTPS development.
Creates certs/ directory and cert.pem + key.pem for use with web_ui.py.
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CERTS_DIR = BASE_DIR / "certs"
CERT_FILE = CERTS_DIR / "cert.pem"
KEY_FILE = CERTS_DIR / "key.pem"


def generate_with_openssl():
    """Use openssl CLI to generate self-signed cert (works on macOS/Linux)."""
    CERTS_DIR.mkdir(exist_ok=True)
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-nodes",
        "-out", str(CERT_FILE),
        "-keyout", str(KEY_FILE),
        "-days", "365",
        "-subj", "/CN=localhost/O=OpenInt/C=US",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"openssl failed: {e}", file=sys.stderr)
        return False


def main():
    if CERT_FILE.exists() and KEY_FILE.exists():
        print(f"Certificates already exist: {CERTS_DIR}")
        print(f"  cert: {CERT_FILE}")
        print(f"  key:  {KEY_FILE}")
        return 0
    print("Generating self-signed TLS certificates for local HTTPS...")
    if generate_with_openssl():
        print(f"Created: {CERT_FILE}, {KEY_FILE}")
        print("Start the server with HTTPS (web_ui.py will use these automatically).")
        return 0
    print("Install OpenSSL or run manually:", file=sys.stderr)
    print(f"  mkdir -p {CERTS_DIR}", file=sys.stderr)
    print(f"  openssl req -x509 -newkey rsa:4096 -nodes -out {CERT_FILE} -keyout {KEY_FILE} -days 365 -subj /CN=localhost", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
