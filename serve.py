#!/usr/bin/env python3
import os
import sys
import subprocess
import json
import shutil
import threading
import itertools
import time
import socket
import ssl
import ipaddress
import urllib.request
from datetime import datetime, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler

# ─── Constants ───────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
VENV_FLAG   = "--in-venv"
VENV_DIR    = os.path.join(SCRIPT_DIR, "venv")
HTTPS_PORT  = 443

# ─── Spinner for long operations ──────────────────────────────────────────────
class Spinner:
    def __init__(self, msg):
        self.msg   = msg
        self.spin  = itertools.cycle("|/-\\")
        self._stop = threading.Event()
        self._thr  = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self._stop.is_set():
            sys.stdout.write(f"\r{self.msg} {next(self.spin)}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " "*(len(self.msg)+2) + "\r")
        sys.stdout.flush()
    def __enter__(self): self._thr.start()
    def __exit__(self, exc_type, exc, tb):
        self._stop.set(); self._thr.join()

# ─── Virtualenv bootstrap ─────────────────────────────────────────────────────
def bootstrap_and_run():
    if VENV_FLAG not in sys.argv:
        if not os.path.isdir(VENV_DIR):
            with Spinner("Creating virtualenv…"):
                subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        pip = os.path.join(VENV_DIR, "Scripts" if os.name=="nt" else "bin", "pip")
        with Spinner("Installing dependencies…"):
            subprocess.check_call([pip, "install", "cryptography"])
        py = os.path.join(VENV_DIR, "Scripts" if os.name=="nt" else "bin", "python")
        os.execv(py, [py, __file__, VENV_FLAG])
    else:
        sys.argv.remove(VENV_FLAG)
        main()

# ─── Config I/O ───────────────────────────────────────────────────────────────
def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            return json.load(open(CONFIG_PATH))
        except:
            pass
    return {"serve_path": os.getcwd()}

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)

# ─── Networking Helpers ──────────────────────────────────────────────────────
def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]; s.close()
    return ip

def get_public_ip():
    try:
        return urllib.request.urlopen("https://api.ipify.org").read().decode().strip()
    except:
        return None

# ─── ASCII Banner ────────────────────────────────────────────────────────────
def print_banner(lan, public):
    lines = [
        f"  Local : https://{lan}",
        f"  Public: https://{public}" if public else "  Public: <none>"
    ]
    w = max(len(l) for l in lines) + 4
    print("\n╔" + "═"*w + "╗")
    for l in lines:
        print("║" + l.ljust(w) + "║")
    print("╚" + "═"*w + "╝\n")

# ─── Certificate Generation ──────────────────────────────────────────────────
def generate_cert(cert_file, key_file):
    lan_ip    = get_lan_ip()
    public_ip = get_public_ip()

    # Always generate a self-signed certificate (no mkcert)
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509 import NameOID, SubjectAlternativeName, DNSName, IPAddress
    import cryptography.x509 as x509
    import ipaddress

    # Generate private key
    keyobj = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Build SAN list
    san_list = [
        DNSName(lan_ip),
        DNSName("localhost"),
        IPAddress(ipaddress.IPv4Address("127.0.0.1"))
    ]
    if public_ip:
        try:
            san_list.append(IPAddress(ipaddress.IPv4Address(public_ip)))
        except ValueError:
            pass

    san = SubjectAlternativeName(san_list)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, lan_ip)])

    # Build and sign certificate
    with Spinner("Generating self-signed certificate…"):
        cert = (
            x509.CertificateBuilder()
               .subject_name(name)
               .issuer_name(name)
               .public_key(keyobj.public_key())
               .serial_number(x509.random_serial_number())
               .not_valid_before(datetime.utcnow())
               .not_valid_after(datetime.utcnow() + timedelta(days=365))
               .add_extension(san, critical=False)
               .sign(keyobj, hashes.SHA256())
        )

    # Write key and cert
    with open(key_file, "wb") as f:
        f.write(keyobj.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

# ─── Main Flow ────────────────────────────────────────────────────────────────
def main():
    # 1) If not root, re-launch under sudo so we can bind :443
    if os.geteuid() != 0:
        print("⚠ Need root to bind port 443; re-running with sudo…")
        os.execvp("sudo", ["sudo", sys.executable] + sys.argv)

    # 2) Load config and ensure required fields exist
    cfg = load_config()
    updated = False
    if not os.path.exists(CONFIG_PATH):
        cfg["serve_path"] = input(f"Serve path [{cfg['serve_path']}]: ") or cfg["serve_path"]
        updated = True
    for key, default in {"serve_path": os.getcwd()}.items():
        if key not in cfg:
            cfg[key] = default
            updated = True
    if updated:
        save_config(cfg)

    # 3) cd into serve directory
    os.chdir(cfg["serve_path"])

    # 4) Generate/load cert.pem & key.pem
    cert_file = os.path.join(os.getcwd(), "cert.pem")
    key_file  = os.path.join(os.getcwd(), "key.pem")
    generate_cert(cert_file, key_file)

    # 5) Build SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)

    # 6) Print LAN & public URLs
    lan_ip    = get_lan_ip()
    public_ip = get_public_ip()
    print_banner(lan_ip, public_ip)

    # 7) Launch Node.js under HTTPS if present
    if os.path.exists("package.json") and os.path.exists("server.js") and shutil.which("node"):
        patch = os.path.join(os.getcwd(), "tls_patch.js")
        with open(patch, "w") as f:
            f.write("""\
const fs = require('fs');
const https = require('https');
const http = require('http');
http.createServer = (opts, listener) => {
  if (typeof opts === 'function') listener = opts;
  return https.createServer({
    key: fs.readFileSync('key.pem'),
    cert: fs.readFileSync('cert.pem')
  }, listener);
};""")
        env = os.environ.copy()
        env["PORT"] = str(HTTPS_PORT)
        cmd = [shutil.which("node"), "-r", patch, "server.js"]
        with Spinner("Starting Node.js (HTTPS on 443)…"):
            proc = subprocess.Popen(cmd, env=env)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    else:
        # Python HTTPS fallback: try primary port, else pick a free port
        import errno
        port = HTTPS_PORT
        for p in range(HTTPS_PORT, HTTPS_PORT + 10):
            try:
                httpd = HTTPServer(("0.0.0.0", p), SimpleHTTPRequestHandler)
            except OSError as e:
                if e.errno == errno.EADDRINUSE:
                    continue
                raise
            else:
                port = p
                break
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        if port != HTTPS_PORT:
            print(f"⚠ Port {HTTPS_PORT} in use; serving on port {port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    bootstrap_and_run()
