#!/usr/bin/env python3
# test_auth.py

"""
Simple script to test API key authentication for job management endpoints.
"""

import os
import requests
import json


def test_auth(base_url: str = "http://localhost:8000"):
    """Test authentication scenarios."""
    jobs_url = f"{base_url}/jobs"

    print("🔐 Testing Job Management API Authentication\n")

    # Test 1: No API key (should fail)
    print("1️⃣ Testing without API key (should fail with 401)...")
    response = requests.get(f"{jobs_url}/status")
    print(f"   Status: {response.status_code}")
    if response.status_code == 401:
        print("   ✅ Correctly rejected - no API key provided")
    else:
        print(f"   ❌ Unexpected response: {response.text}")

    # Test 2: Invalid API key (should fail)
    print("\n2️⃣ Testing with invalid API key (should fail with 403)...")
    headers = {"Authorization": "Bearer invalid-key"}
    response = requests.get(f"{jobs_url}/status", headers=headers)
    print(f"   Status: {response.status_code}")
    if response.status_code == 403:
        print("   ✅ Correctly rejected - invalid API key")
    else:
        print(f"   ❌ Unexpected response: {response.text}")

    # Test 3: Valid API key (should succeed if configured)
    api_key = os.environ.get("SECRET_API_KEY")
    if api_key:
        print("\n3️⃣ Testing with valid API key (should succeed)...")
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{jobs_url}/status", headers=headers)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Authentication successful!")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"   ❌ Authentication failed: {response.text}")
    else:
        print("\n3️⃣ Skipping valid API key test - SECRET_API_KEY not set")
        print("   Set SECRET_API_KEY environment variable to test valid authentication")

    print("\n🏁 Authentication tests completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test job management API authentication"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the news crawler service",
    )

    args = parser.parse_args()

    try:
        test_auth(base_url=args.url)
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the news crawler service is running")
    except Exception as e:
        print(f"❌ Test failed: {e}")
