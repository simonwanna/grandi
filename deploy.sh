#!/bin/bash
# Environment agnostic deployment script
set -e  # Exit on error

# Load environment variables from .env if it exists (for local development)
if [ -f ".env" ]; then
  # Automatically export variables defined in .env
  set -a
  source ".env"
  set +a
fi

# Ensure required variables are set (either from .env or CI/CD secrets)
if [ -z "$GCP_PROJECT_ID" ]; then
  echo "Error: GCP_PROJECT_ID is not set."
  echo "Please set it in .env or as an environment variable."
  exit 1
fi

if [ -z "$BUCKET_NAME" ]; then
  echo "Error: BUCKET_NAME is not set."
  echo "Please set it in .env or as an environment variable."
  exit 1
fi

echo "Deploying to Project: $GCP_PROJECT_ID"
echo "Using Bucket: $BUCKET_NAME"

# Authentication: use GCP_SA_KEY (can be a file path or JSON content)
if [ -n "$GCP_SA_KEY" ]; then
  if [ -f "$GCP_SA_KEY" ]; then
    echo "Authenticating using key file at $GCP_SA_KEY..."
    gcloud auth activate-service-account --key-file="$GCP_SA_KEY"
  else
    echo "Authenticating using GCP_SA_KEY environment variable..."
    echo "$GCP_SA_KEY" > /tmp/gcp_key.json
    gcloud auth activate-service-account --key-file=/tmp/gcp_key.json
  fi
else
  echo "No service account key found. Assuming already authenticated..."
fi

# Enable required services (idempotent)
echo "Enabling required GCP services..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com --project "$GCP_PROJECT_ID"

# Build the container image
echo "Submitting build..."
gcloud builds submit --project "$GCP_PROJECT_ID" --tag eu.gcr.io/"$GCP_PROJECT_ID"/chess-inference app

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy chess-inference \
  --project "$GCP_PROJECT_ID" \
  --image eu.gcr.io/"$GCP_PROJECT_ID"/chess-inference \
  --platform managed \
  --region europe-north1 \
  --memory 2Gi \
  --set-env-vars BUCKET_NAME="$BUCKET_NAME"

echo "Deployment complete."
