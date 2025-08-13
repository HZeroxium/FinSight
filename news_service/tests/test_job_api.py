#!/usr/bin/env python3
# test_job_api.py

"""
Simple test script for job management API endpoints.
Run this after starting the news crawler service to test the job management endpoints.
"""

import os
import requests
import json
import time
from typing import Dict, Any


class JobAPITester:
    """Test class for job management API endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        """
        Initialize the API tester.

        Args:
            base_url: Base URL of the news crawler service
            api_key: Secret API key for admin endpoints
        """
        self.base_url = base_url
        self.jobs_url = f"{base_url}/jobs"
        self.headers = {}

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def test_job_info(self) -> Dict[str, Any]:
        """Test job service info endpoint."""
        print("🔍 Testing job service info...")
        response = requests.get(f"{self.jobs_url}/", headers=self.headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return response.json()
        else:
            print(f"Error: {response.text}")
            return {"error": response.text}

    def test_job_status(self) -> Dict[str, Any]:
        """Test job status endpoint."""
        print("\n📊 Testing job status...")
        response = requests.get(f"{self.jobs_url}/status", headers=self.headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return response.json()
        else:
            print(f"Error: {response.text}")
            return {"error": response.text}

    def test_job_config(self) -> Dict[str, Any]:
        """Test job configuration endpoints."""
        print("\n⚙️ Testing job configuration...")

        # Get current config
        response = requests.get(f"{self.jobs_url}/config", headers=self.headers)
        print(f"Get Config Status: {response.status_code}")
        if response.status_code == 200:
            config = response.json()
            print(f"Current Config: {json.dumps(config, indent=2)}")
            return config
        else:
            print(f"Error: {response.text}")
            return {"error": response.text}

    def test_update_config(self) -> Dict[str, Any]:
        """Test updating job configuration."""
        print("\n🔧 Testing config update...")

        update_data = {
            "max_items_per_source": 25,
            "schedule": "0 */3 * * *",  # Every 3 hours
        }

        response = requests.put(
            f"{self.jobs_url}/config", json=update_data, headers=self.headers
        )
        print(f"Update Config Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Update Response: {json.dumps(response.json(), indent=2)}")
            return response.json()
        else:
            print(f"Error: {response.text}")
            return {"error": response.text}

    def test_job_stats(self) -> Dict[str, Any]:
        """Test job statistics endpoint."""
        print("\n📈 Testing job statistics...")
        response = requests.get(f"{self.jobs_url}/stats", headers=self.headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return response.json()
        else:
            print(f"Error: {response.text}")
            return {"error": response.text}

    def test_manual_job(self) -> Dict[str, Any]:
        """Test manual job execution."""
        print("\n🔄 Testing manual job execution...")

        job_data = {
            "sources": ["coindesk"],
            "max_items_per_source": 5,
            "config_overrides": {},
        }

        response = requests.post(
            f"{self.jobs_url}/run", json=job_data, headers=self.headers
        )
        print(f"Manual Job Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Manual Job Response: {json.dumps(response.json(), indent=2)}")
            return response.json()
        else:
            print(f"Error: {response.text}")
            return {"error": response.text}

    def test_job_health(self) -> Dict[str, Any]:
        """Test job health check."""
        print("\n💚 Testing job health check...")
        response = requests.get(f"{self.jobs_url}/health", headers=self.headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return response.json()
        else:
            print(f"Error: {response.text}")
            return {"error": response.text}

    def test_start_job(self) -> Dict[str, Any]:
        """Test starting the job service."""
        print("\n🚀 Testing job start...")

        start_data = {"force_restart": False}

        response = requests.post(
            f"{self.jobs_url}/start", json=start_data, headers=self.headers
        )
        print(f"Start Job Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Start Job Response: {json.dumps(response.json(), indent=2)}")
            return response.json()
        else:
            print(f"Error: {response.text}")
            return {"error": response.text}

    def test_stop_job(self) -> Dict[str, Any]:
        """Test stopping the job service."""
        print("\n🛑 Testing job stop...")

        stop_data = {"graceful": True}

        response = requests.post(
            f"{self.jobs_url}/stop", json=stop_data, headers=self.headers
        )
        print(f"Stop Job Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Stop Job Response: {json.dumps(response.json(), indent=2)}")
            return response.json()
        else:
            print(f"Error: {response.text}")
            return {"error": response.text}

    def run_basic_tests(self):
        """Run basic non-destructive tests."""
        print("🧪 Running basic job management API tests...\n")

        try:
            # Test basic info and status endpoints
            self.test_job_info()
            self.test_job_status()
            self.test_job_config()
            self.test_job_stats()
            self.test_job_health()

            print("\n✅ Basic tests completed successfully!")

        except requests.exceptions.ConnectionError:
            print("❌ Connection error: Make sure the news crawler service is running")
        except Exception as e:
            print(f"❌ Test failed: {e}")

    def run_full_tests(self, include_destructive: bool = False):
        """Run full test suite including manual job execution."""
        print("🧪 Running full job management API tests...\n")

        try:
            # Run basic tests first
            self.run_basic_tests()

            # Test manual job execution
            print("\n" + "=" * 50)
            self.test_manual_job()

            if include_destructive:
                print("\n" + "=" * 50)
                print("⚠️  Running destructive tests (start/stop)...")
                self.test_start_job()
                time.sleep(2)
                self.test_stop_job()

            print("\n✅ All tests completed!")

        except requests.exceptions.ConnectionError:
            print("❌ Connection error: Make sure the news crawler service is running")
        except Exception as e:
            print(f"❌ Test failed: {e}")


def main():
    """Main function to run the tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test job management API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the news crawler service",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full tests including manual job execution",
    )
    parser.add_argument(
        "--destructive",
        action="store_true",
        help="Include destructive tests (start/stop operations)",
    )
    parser.add_argument(
        "--api-key",
        help="Secret API key for admin endpoints (can also use SECRET_API_KEY env var)",
    )

    args = parser.parse_args()

    # Get API key from command line or environment variable
    api_key = args.api_key or os.environ.get("SECRET_API_KEY")
    if not api_key:
        print(
            "⚠️ Warning: No API key provided. Use --api-key or set SECRET_API_KEY environment variable"
        )
        print("   Tests will likely fail with 401/403 errors")

    tester = JobAPITester(base_url=args.url, api_key=api_key)

    if args.full:
        tester.run_full_tests(include_destructive=args.destructive)
    else:
        tester.run_basic_tests()


if __name__ == "__main__":
    main()
