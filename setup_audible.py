#!/usr/bin/env python3
"""Complete Audible authentication using the audible library directly."""
import readline  # fixes macOS paste issues
import audible
import sys
from pathlib import Path

AUDIBLE_DIR = Path.home() / ".audible"
AUTH_FILE = AUDIBLE_DIR / "audible.json"
CONFIG_FILE = AUDIBLE_DIR / "config.toml"

print("Starting Audible external browser login...")
print("A browser window will open. Log in with your Amazon credentials.")
print("After login you'll see a 'Page not found' — that's expected.")
print("Copy the URL from the address bar and paste it here.\n")

auth = audible.Authenticator.from_login_external(locale="us")

AUDIBLE_DIR.mkdir(parents=True, exist_ok=True)
auth.to_file(str(AUTH_FILE))
print(f"\nAuth saved to: {AUTH_FILE}")

# Create config.toml if it doesn't exist
if not CONFIG_FILE.exists():
    CONFIG_FILE.write_text(
        '[profile.audible]\n'
        'auth_file = "audible.json"\n'
        'country_code = "us"\n'
        'is_primary = true\n'
    )
    print(f"Config saved to: {CONFIG_FILE}")

print("\nDone! You can now use: audible activation-bytes")
