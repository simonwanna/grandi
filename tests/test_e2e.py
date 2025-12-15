import os

import dotenv
import httpx
import pytest

dotenv.load_dotenv()

SERVICE_URL = os.getenv("SERVICE_URL")


@pytest.mark.skipif(not SERVICE_URL, reason="SERVICE_URL not set")
def test_predict_endpoint() -> None:
    """
    End-to-End test for the deployed prediction endpoint.
    Sends a sample payload and verifies the response.
    """
    url = f"{SERVICE_URL.rstrip('/')}/predict"

    # Get Google Identity Token (Required because the service is Private/IAM-protected)
    try:
        import subprocess

        print("Fetching GCloud Identity Token...")
        id_token = subprocess.check_output(["gcloud", "auth", "print-identity-token"], text=True).strip()
    except Exception as e:
        pytest.fail(f"Failed to get gcloud token: {e}")

    # Authenticate with Authorization: Bearer <ID_TOKEN> (Unlocks Cloud Run)
    headers = {"Authorization": f"Bearer {id_token}"}

    payload = [0.5] * 64

    print(f"Testing URL: {url} with payload size: {len(payload)}")

    response = httpx.post(url, json=payload, headers=headers, timeout=10.0)

    # Assertions
    assert response.status_code == 200, f"Request failed with {response.status_code}: {response.text}"

    data = response.json()
    assert "win_prob" in data, "Response JSON missing 'win_prob' key"
    assert isinstance(data["win_prob"], float), "'win_prob' should be a float"
    assert 0.0 <= data["win_prob"] <= 1.0, "'win_prob' should be between 0 and 1"

    print(f"Response: {data}")
