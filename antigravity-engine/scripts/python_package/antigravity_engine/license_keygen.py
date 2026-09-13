#!/usr/bin/env python3
"""Antigravity Engine — Ed25519 License Key Generator

Usage:
  python3 license_keygen.py init          # Generate keypair (first time only)
  python3 license_keygen.py generate      # Generate a license key
    --licensee "Clinic Name" \
    --features clinical,sdk \
    --max-channels 8 \
    --expiry 2027-12-31 \
    --device-hash "abc123"  # optional
  python3 license_keygen.py verify KEY    # Verify a license key
  python3 license_keygen.py pubkey        # Print public key bytes for embedding
"""

import argparse
import base64
import json
import os
import sys
from datetime import datetime
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

KEY_PATH = os.path.expanduser("~/.antigravity/license_private_key.pem")

def get_private_key():
    if not os.path.exists(KEY_PATH):
        print(f"Error: Private key not found at {KEY_PATH}")
        print("Run 'python3 license_keygen.py init' first.")
        sys.exit(1)
    
    with open(KEY_PATH, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=None
        )
    return private_key

def init_cmd(args):
    if os.path.exists(KEY_PATH):
        print(f"Key already exists at {KEY_PATH}")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(KEY_PATH), exist_ok=True)
    
    private_key = ed25519.Ed25519PrivateKey.generate()
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    with open(KEY_PATH, "wb") as f:
        f.write(pem)
        
    os.chmod(KEY_PATH, 0o600)
    print(f"Generated new Ed25519 private key at {KEY_PATH}")
    pubkey_cmd(args)

def generate_cmd(args):
    private_key = get_private_key()
    
    if args.perpetual:
        expiry_epoch = 0
    elif args.expiry:
        dt = datetime.strptime(args.expiry, "%Y-%m-%d")
        expiry_epoch = int(dt.timestamp())
    else:
        print("Error: Must specify either --expiry or --perpetual")
        sys.exit(1)
        
    features = [f.strip() for f in args.features.split(",")] if args.features else []
    
    payload = {
        "licensee": args.licensee,
        "expiryEpoch": expiry_epoch,
        "features": features,
        "maxChannels": args.max_channels,
    }
    
    if args.device_hash:
        payload["deviceHash"] = args.device_hash
        
    payload_json = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    payload_b64 = base64.urlsafe_b64encode(payload_json).decode('ascii').rstrip('=')
    
    signature = private_key.sign(payload_json)
    signature_b64 = base64.urlsafe_b64encode(signature).decode('ascii').rstrip('=')
    
    license_key = f"{payload_b64}.{signature_b64}"
    print(license_key)

def verify_cmd(args):
    private_key = get_private_key()
    public_key = private_key.public_key()
    
    try:
        parts = args.key.split('.')
        if len(parts) != 2:
            raise ValueError("Invalid format: expected exactly one dot")
            
        payload_b64 = parts[0] + "=" * (-len(parts[0]) % 4)
        sig_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        
        payload_json = base64.urlsafe_b64decode(payload_b64)
        signature = base64.urlsafe_b64decode(sig_b64)
        
        public_key.verify(signature, payload_json)
        
        payload = json.loads(payload_json)
        print("Signature Valid!")
        print(json.dumps(payload, indent=2))
        
    except Exception as e:
        print(f"Verification Failed: {e}")
        sys.exit(1)

def pubkey_cmd(args):
    private_key = get_private_key()
    public_key = private_key.public_key()
    raw_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    
    print("Swift public key bytes:")
    swift_array = ", ".join(f"0x{b:02x}" for b in raw_bytes)
    print(f"[{swift_array}]")
    print("\nC++ public key bytes:")
    cpp_array = ", ".join(f"0x{b:02x}" for b in raw_bytes)
    print(f"uint8_t pubkey[] = {{{cpp_array}}};")

def main():
    parser = argparse.ArgumentParser(description="Antigravity Engine License Keygen")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("init", help="Generate new private key")
    
    gen_parser = subparsers.add_parser("generate", help="Generate a license key")
    gen_parser.add_argument("--licensee", required=True, help="Licensee name")
    gen_parser.add_argument("--features", required=True, help="Comma-separated features")
    gen_parser.add_argument("--max-channels", type=int, required=True, help="Max parallel channels")
    group = gen_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--expiry", help="Expiry date YYYY-MM-DD")
    group.add_argument("--perpetual", action="store_true", help="Perpetual license (no expiry)")
    gen_parser.add_argument("--device-hash", help="Optional device hash binding")
    
    verify_parser = subparsers.add_parser("verify", help="Verify a license key")
    verify_parser.add_argument("key", help="The license key string")
    
    subparsers.add_parser("pubkey", help="Print public key as byte arrays")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_cmd(args)
    elif args.command == "generate":
        generate_cmd(args)
    elif args.command == "verify":
        verify_cmd(args)
    elif args.command == "pubkey":
        pubkey_cmd(args)

if __name__ == "__main__":
    main()
