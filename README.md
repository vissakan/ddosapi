#  Adaptive Application-Layer DDoS Mitigation using Deep Reinforcement Learning

> **An intelligent, cost-aware DDoS defense framework powered by Deep Recurrent Q-Networks (DRQN), deployed as a production-ready FastAPI microservice with full CI/CD automation.**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

##  Table of Contents

1. [Problem Statement](#-problem-statement)
2. [How It Works — Conceptual Overview](#-how-it-works--conceptual-overview)
3. [System Architecture](#-system-architecture)
4. [Solution Design](#-solution-design)
5. [Reinforcement Learning Framework](#-reinforcement-learning-framework)
6. [Model Architecture](#-model-architecture)
7. [API Reference](#-api-reference)
8. [Project Structure](#-project-structure)
9. [Getting Started](#-getting-started)
10. [CI/CD Pipeline](#-cicd-pipeline)
11. [Key Results](#-key-results)
12. [Limitations & Future Work](#-limitations--future-work)
13. [Author](#-author)

---

##  Problem Statement

Traditional DDoS mitigation systems depend on **static rules** or **signature-based detection**. These approaches fail against modern **application-layer (Layer 7) attacks** because:

- Attack traffic closely resembles legitimate HTTP/S requests
- Packet-level inspection cannot distinguish intent
- Static rules cannot adapt to evolving attack patterns
- Over-blocking harms real users; under-blocking harms availability

```
Traditional Approach:
┌─────────────────────────────────────────────────────┐
│  Incoming Traffic                                   │
│       ↓                                             │
│  Static Rule Engine  ──→  Block / Allow (Fixed)     │
│                                                     │
│     Cannot adapt to new patterns                    │
│     High false positive rate                        │
│     Ignores temporal context                        │
└─────────────────────────────────────────────────────┘

This Project's Approach:
┌─────────────────────────────────────────────────────┐
│  Incoming Traffic + Real-Time Metrics               │
│       ↓                                             │
│  DRQN Agent  ──→  Adaptive Mitigation Action        │
│       ↑                                             │
│  Learns from environment feedback (reward signal)   │
│                                                     │
│     Adapts to unseen attack patterns                │
│     Balances security cost vs. user impact          │
│     Retains temporal context via LSTM memory        │
└─────────────────────────────────────────────────────┘
```

---

##  How It Works — Conceptual Overview

This system models DDoS mitigation as a **sequential decision-making problem** under **partial observability**, solved using **Deep Reinforcement Learning (DRL)**.

At each timestep, the agent:

1. **Observes** application-layer metrics (request rate, latency, CPU usage, etc.)
2. **Decides** a mitigation action (e.g., rate-limit, CAPTCHA, block, or allow)
3. **Receives** a reward signal reflecting both security effectiveness and cost
4. **Updates** its policy to maximize long-term cumulative reward

```
                     ┌──────────────────────────────┐
                     │         Environment           │
                     │   (Application-Layer Server)  │
                     └──────────┬───────────┬────────┘
                                │           │
                    State sₜ   │           │  Reward rₜ
                   (Metrics)   ↓           ↓
                     ┌──────────────────────────────┐
                     │         DRQN Agent            │
                     │  ┌────────────────────────┐  │
                     │  │  LSTM Hidden State hₜ  │  │
                     │  │  (Temporal Memory)      │  │
                     │  └────────────────────────┘  │
                     │  ┌────────────────────────┐  │
                     │  │  Q-Value Estimation     │  │
                     │  │  Q(s, a | h)            │  │
                     │  └────────────────────────┘  │
                     └──────────────┬───────────────┘
                                    │
                         Action aₜ  │
                       (Mitigation) ↓
                     ┌──────────────────────────────┐
                     │     Mitigation Actuator       │
                     │  [Allow | Rate-Limit |        │
                     │   CAPTCHA | Throttle | Block] │
                     └──────────────────────────────┘
```

---

##  System Architecture

The full system spans training, deployment, and CI/CD automation across three layers:

```
╔══════════════════════════════════════════════════════════════════════╗
║                        SYSTEM ARCHITECTURE                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   ┌──────────────────────────────────────────────────────────────┐  ║
║   │                   TRAINING LAYER                             │  ║
║   │                                                              │  ║
║   │   Simulated DDoS        Reward      DQN Baseline             │  ║
║   │   Environment    ──→   Designer ──→ / DRQN Agent             │  ║
║   │                                         │                    │  ║
║   │                                    model.pth                 │  ║
║   └──────────────────────────────────────┬───────────────────────┘  ║
║                                          │ Trained Weights           ║
║   ┌──────────────────────────────────────▼───────────────────────┐  ║
║   │                   INFERENCE LAYER                            │  ║
║   │                                                              │  ║
║   │   HTTP Request                                               │  ║
║   │   ──────────→  FastAPI (main.py)                             │  ║
║   │                     │                                        │  ║
║   │                     ▼                                        │  ║
║   │               Inference Wrapper (model.py)                   │  ║
║   │                     │                                        │  ║
║   │                     ▼                                        │  ║
║   │               DRQN Architecture (drqn_model.py)              │  ║
║   │                     │                                        │  ║
║   │                     ▼                                        │  ║
║   │              Mitigation Action Response                      │  ║
║   └──────────────────────────────────────┬───────────────────────┘  ║
║                                          │ Containerized             ║
║   ┌──────────────────────────────────────▼───────────────────────┐  ║
║   │                   DEPLOYMENT LAYER                           │  ║
║   │                                                              │  ║
║   │   Git Push → GitHub Actions → Tests → Docker Build → Push   │  ║
║   │                                                              │  ║
║   │   DockerHub: ddos-api:latest  /  ddos-api:<commit-sha>       │  ║
║   └──────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

##  Solution Design

### Application-Layer Simulation Environment

The custom environment simulates real-world conditions including benign traffic, DDoS floods, and mixed scenarios.

```
┌─────────────────────────────────────────────────────────────────┐
│                   SIMULATION ENVIRONMENT                        │
│                                                                 │
│  Traffic Generator                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Benign Users │    │  DDoS Bots   │    │  Mixed Scenario  │  │
│  │  (Low rate)  │    │  (High rate) │    │  (Blended)       │  │
│  └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘  │
│         │                   │                     │             │
│         └─────────────┬─────┘─────────────────────┘            │
│                       ↓                                         │
│              Server Metric Collector                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  request_rate │ error_rate │ latency │ queue_len │ cpu   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                       ↓                                         │
│             Observation Vector  →  Agent                        │
│                       ↑                                         │
│             Action from Agent   ←  Mitigation Applied          │
│                       ↓                                         │
│               Reward Calculation                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  + Legitimate requests served                            │  │
│  │  - Attack requests that passed through                   │  │
│  │  - Mitigation cost (per action type)                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Objective Reward Design

The reward function balances **security effectiveness** against **operational cost**:

```
R(t) = α · (Legitimate Requests Served)
     - β · (Attack Requests Allowed)
     - γ · (Action Cost)

Where:
  α = reward weight for serving legitimate traffic
  β = penalty weight for missed attacks
  γ = cost coefficient per action type

Action Costs:
  ┌─────────────────────────────────┐
  │ 0: Allow        → Cost = 0.0   │
  │ 1: Rate-Limit   → Cost = 0.1   │
  │ 2: CAPTCHA      → Cost = 0.3   │
  │ 3: Throttle     → Cost = 0.4   │
  │ 4: Block        → Cost = 0.5   │
  └─────────────────────────────────┘
```

---

##  Reinforcement Learning Framework

### From DQN to DRQN

| Feature | DQN (Baseline) | DRQN (This Project) |
|---|---|---|
| Memory | ❌ None (Markov assumption) | ✅ LSTM hidden state |
| Observability | Full state required | Handles partial observability |
| Temporal context | ❌ Single timestep | ✅ Sequential timesteps |
| Attack detection | Reactive | Predictive (pattern-aware) |

### Training Loop

```
Initialize DRQN, Replay Buffer, Target Network
         │
         ▼
┌────────────────────────────────────────────────────┐
│                  TRAINING EPISODE                  │
│                                                    │
│  Reset Environment  →  Initial Observation s₀      │
│          │                                         │
│          ▼                                         │
│  ┌───────────────────────────────────────────┐    │
│  │              TIMESTEP LOOP                │    │
│  │                                           │    │
│  │  1. Select action aₜ                     │    │
│  │     ε-greedy: explore or exploit          │    │
│  │                                           │    │
│  │  2. Apply action to environment           │    │
│  │     → Receive sₜ₊₁, rₜ, done            │    │
│  │                                           │    │
│  │  3. Store (sₜ, aₜ, rₜ, sₜ₊₁) in buffer │    │
│  │                                           │    │
│  │  4. Sample mini-batch from buffer         │    │
│  │                                           │    │
│  │  5. Compute TD Target:                    │    │
│  │     y = rₜ + γ · max Q_target(sₜ₊₁, a') │    │
│  │                                           │    │
│  │  6. Update DRQN weights via MSE loss      │    │
│  │     L = (y - Q(sₜ, aₜ))²                │    │
│  │                                           │    │
│  │  7. Periodically sync Target Network      │    │
│  └───────────────────────────────────────────┘    │
│          │                                         │
│          ▼                                         │
│     Decay ε  →  Next Episode                       │
└────────────────────────────────────────────────────┘
         │
         ▼
    Save model.pth
```

---

##  Model Architecture

```
Input: Observation Vector  [5 features per timestep]
         │
         ▼
┌────────────────────────────────────────────┐
│         Fully Connected Layer              │
│         Linear(5 → 64) + ReLU             │
└────────────────────────────┬───────────────┘
                             │
                             ▼
┌────────────────────────────────────────────┐
│           LSTM Layer                       │
│         LSTM(64 → 128)                     │
│   Retains hidden state hₜ across steps     │
│   Enables temporal pattern recognition     │
└────────────────────────────┬───────────────┘
                             │
                             ▼
┌────────────────────────────────────────────┐
│         Fully Connected Layer              │
│         Linear(128 → 64) + ReLU           │
└────────────────────────────┬───────────────┘
                             │
                             ▼
┌────────────────────────────────────────────┐
│           Q-Value Output Layer             │
│           Linear(64 → num_actions)         │
│   Outputs Q(s, a) for each action          │
└────────────────────────────┬───────────────┘
                             │
                             ▼
              argmax → Mitigation Action
```

---

## 🌐 API Reference

The trained DRQN model is served as a REST API via **FastAPI**.

### Base URL
```
http://localhost:8000
```

### Endpoints

#### `POST /predict`
Runs inference and returns the recommended mitigation action.

**Request Body:**
```json
{
  "observation": [request_rate, error_rate, latency, queue_length, cpu_usage]
}
```

| Field | Type | Description |
|---|---|---|
| `request_rate` | float | Incoming requests per second |
| `error_rate` | float | Fraction of requests returning errors |
| `latency` | float | Average response latency (ms) |
| `queue_length` | float | Current request queue depth |
| `cpu_usage` | float | Server CPU utilization (0.0–1.0) |

**Response:**
```json
{
  "mitigation_action": 2
}
```

**Action Mapping:**

| Value | Action | Description |
|---|---|---|
| `0` | Allow | No mitigation — normal traffic |
| `1` | Rate-Limit | Limit requests per IP |
| `2` | CAPTCHA | Challenge suspicious clients |
| `3` | Throttle | Slow response delivery |
| `4` | Block | Hard-block source IPs |

#### `GET /health`
```json
{ "status": "ok" }
```

#### `GET /docs`
Interactive Swagger UI — available at `http://localhost:8000/docs`

---

## 📂 Project Structure

```
ddos-mitigation-drl/
│
├── app/
│   ├── main.py              # FastAPI service — defines API endpoints
│   ├── model.py             # Model loading & inference wrapper
│   ├── drqn_model.py        # DRQN neural network architecture (PyTorch)
│   └── model.pth            # Trained model weights
│
├── tests/
│   ├── test_api.py          # API endpoint integration tests
│   └── test_model.py        # Model inference unit tests
│
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI/CD pipeline definition
│
├── Dockerfile               # Container build instructions
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

##  Getting Started

### Prerequisites

- Docker 20.x+
- Python 3.10 (for local development)
- Git

### Option 1 — Run with Docker (Recommended)

**Step 1: Clone the repository**
```bash
git clone https://github.com/<your-username>/ddos-mitigation-drl.git
cd ddos-mitigation-drl
```

**Step 2: Build the Docker image**
```bash
docker build -t ddos-api .
```

**Step 3: Run the container**
```bash
docker run -p 8000:8000 ddos-api
```

**Step 4: Test the API**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"observation": [1200.0, 0.45, 320.0, 85.0, 0.91]}'
```

**Step 5: Open the interactive docs**
```
http://localhost:8000/docs
```

### Option 2 — Run Locally (Python)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run Tests

```bash
pytest tests/ -v
```

---

##  CI/CD Pipeline

Every push to the `main` branch automatically triggers the following pipeline:

```
Developer Pushes to main
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                  GitHub Actions                         │
│                                                         │
│  Step 1: Checkout Repository                            │
│          └─→ Clone source code                          │
│                                                         │
│  Step 2: Set Up Python 3.10                             │
│          └─→ Install dependencies from requirements.txt │
│                                                         │
│  Step 3: Run Automated Tests                            │
│          └─→ pytest tests/                              │
│               ├─ test_api.py    (API integration tests) │
│               └─ test_model.py  (Model unit tests)      │
│                        │                                │
│               ✅ All pass  /  ❌ Fail → Pipeline stops  │
│                                                         │
│  Step 4: Build Docker Image                             │
│          └─→ docker build -t ddos-api .                 │
│                                                         │
│  Step 5: Tag Image with Commit SHA                      │
│          └─→ ddos-api:<commit-sha>                      │
│                                                         │
│  Step 6: Push to DockerHub                              │
│          └─→ ddos-api:latest                            │
│          └─→ ddos-api:<commit-sha>                      │
└─────────────────────────────────────────────────────────┘
         │
         ▼
  Versioned Image Available on DockerHub
```

**Example image tags on DockerHub:**
```
username/ddos-api:latest
username/ddos-api:3f2a8c1
username/ddos-api:a91bc4d
```

The commit SHA tag ensures **full traceability** — every deployed image maps to an exact code state.

---

##  Key Results

| Metric | Observation |
|---|---|
| Reward Convergence | Stable reward curve achieved within training episodes |
| Response Latency | Bounded latency maintained under simulated attack load |
| F1 Score | Progressive improvement across training epochs |
| Mitigation Behavior | Agent learned cost-proportionate escalation |
| Action Distribution | Preference for low-cost actions under benign traffic |

### Agent Behavior Pattern

```
Attack Intensity vs. Agent Response
─────────────────────────────────────────────
Low Attack      →  Allow / Rate-Limit
Medium Attack   →  CAPTCHA / Throttle
High Attack     →  Block
─────────────────────────────────────────────
Agent avoids over-blocking (costly) during
low-intensity phases, escalating response
only when justified by the reward signal.
```

---

##  Limitations & Future Work

### Current Limitations

| Limitation | Details |
|---|---|
| Simulated environment | Trained on synthetic traffic; not validated on real-world captures |
| Single-server setting | No distributed or multi-node attack modeling |
| Manual reward tuning | Reward weights (α, β, γ) require domain expertise to configure |

### Roadmap

```
Phase 1 (Current)
└─→ Single-server DRQN with simulated environment 

Phase 2 (Planned)
└─→ Real traffic dataset integration (CAIDA, CIC-DDoS2019)
└─→ Production monitoring integration (Prometheus + Grafana)

Phase 3 (Research)
└─→ Multi-server deployment
└─→ Multi-agent reinforcement learning for distributed defense

Phase 4 (MLOps)
└─→ Automated model retraining on drift detection
└─→ Shadow deployment and A/B evaluation pipelines
```

---

##  Author

**Vissakan V**
M.Tech — Computer Science and Engineering

*Specializations: Deep Reinforcement Learning · ML Deployment · DevOps*

---

> **Note:** This project was developed as an academic research prototype. The simulation environment and results demonstrate the viability of DRL-based adaptive mitigation, and are not a production security system without further validation against real-world traffic datasets.
