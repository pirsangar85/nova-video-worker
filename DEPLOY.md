# Deploy Your Own Video Model on RunPod

## Cost: ~$0.01-0.03 per video (GPU time only)

## Step 1: Build Docker Image

You need Docker installed. Run from this folder:

```bash
docker build -t your-dockerhub-username/nova-video-worker:latest .
docker push your-dockerhub-username/nova-video-worker:latest
```

Or use RunPod's Docker build (no local Docker needed):
- Go to https://github.com - create a repo, upload handler.py and Dockerfile
- Go to https://hub.docker.com - link to GitHub, auto-build

## Step 2: Create Serverless Endpoint on RunPod

1. Go to https://www.runpod.io/console/serverless
2. Click "New Endpoint"
3. Settings:
   - Docker Image: `your-dockerhub-username/nova-video-worker:latest`
   - GPU: **A100 80GB** (needed for 14B model) or **A40 48GB** (for smaller models)
   - Min Workers: 0 (scale to zero when idle = no cost)
   - Max Workers: 1-3 (scale up for multiple users)
   - Idle Timeout: 30 seconds
   - Flash Boot: ON (faster cold starts)
4. Click "Create"
5. Copy the Endpoint ID

## Step 3: Connect to NOVA App

Open NOVA → Settings → RunPod Endpoint
Paste: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/`

## Cost Breakdown
- A100 80GB: ~$0.0012/sec = $0.07/min
- Video generation takes ~30-60 sec = $0.02-0.04/video
- When idle (no requests): $0.00 (scales to zero)
- Monthly (100 videos/day): ~$60-120/month
- Revenue from 1 Pro user ($14.99/month): Covers ~5-7 users' generation costs

## Alternative: Smaller Model (Cheaper)
Change the model in handler.py to:
- `Wan-AI/Wan2.1-T2V-1.3B` — runs on A40 48GB ($0.0008/sec)
- Cost per video: ~$0.01-0.02
- Quality: Good but not as detailed as 14B
