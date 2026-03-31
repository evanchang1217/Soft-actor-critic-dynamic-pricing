# Soft Actor-Critic Dynamic Pricing

This repository contains an NCKU CSIE capstone project that applies Soft Actor-Critic (SAC) reinforcement learning to dynamic pricing in a simulated in-game economy.

The project combines:
- A PyTorch-based SAC training loop
- A Django web interface for configuring experiments
- A simple dashboard for reward, CPI, and per-product statistics

Project note:
- 2nd place out of 87 teams in the NCKU CSIE capstone competition
- The repository has been cleaned up so reviewers can run it locally more easily

## Features

- Dynamic pricing with Soft Actor-Critic
- Multi-product virtual economy simulation
- Configurable training parameters from a web UI
- Reward and CPI trend visualization
- Optional server status page for a local market API

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Create a local `.env` file from `.env.example`

```env
SECRET_KEY=replace-with-a-local-django-secret-key
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost
```

3. Run database migrations

```bash
python manage.py migrate
```

4. Start the development server

```bash
python manage.py runserver
```

5. Open the app in your browser

- Training page: `http://127.0.0.1:8000/`
- Training page alias: `http://127.0.0.1:8000/start_training/`
- Optional server status page: `http://127.0.0.1:8000/server_status/`

## Notes

- `server_status/` depends on a separate local API running at `127.0.0.1:4567`.
- Training will generate `statics/images/sac_reward.png` locally while the app is running.
- The `Poster/` folder contains the project poster used for presentation.
