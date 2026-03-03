🛡️ Adaptive Application-Layer DDoS Mitigation using Deep Reinforcement Learning
📖 Overview

This project implements an adaptive, cost-aware mitigation framework for application-layer Distributed Denial-of-Service (DDoS) attacks using Deep Reinforcement Learning (DRL).

The system models DDoS mitigation as a sequential decision-making problem under partial observability and deploys a trained Deep Recurrent Q-Network (DRQN) model as a FastAPI microservice. The project includes CI/CD automation, Docker containerization, and automated versioned image deployment to DockerHub.

🎯 Problem Statement

Traditional DDoS mitigation systems rely on static rules or signature-based detection. However, application-layer attacks often resemble legitimate traffic and cannot be reliably distinguished using packet-level inspection alone.

This project addresses:

Adaptive mitigation under partial observability

Cost-aware defensive actions

Real-time inference using application-level metrics

Automated deployment with CI/CD

🧠 Solution Architecture
Monitoring Metrics
        ↓
FastAPI Inference Service
        ↓
Trained DRQN Model
        ↓
Mitigation Action Decision
CI/CD Pipeline
Git Push
   ↓
GitHub Actions
   ↓
Run Tests
   ↓
Build Docker Image
   ↓
Push Versioned Image to DockerHub
🚀 Features

Deep Q-Network (DQN) baseline

Deep Recurrent Q-Network (DRQN)

Multi-objective reward design

Application-layer simulation environment

FastAPI deployment

Docker containerization

Automated CI/CD pipeline

Versioned Docker image builds

🏗️ Tech Stack

Python 3.10

PyTorch

FastAPI

Docker

GitHub Actions (CI/CD)

Pytest (Automated testing)

📂 Project Structure
app/
 ├── main.py          # FastAPI service
 ├── model.py         # Model loading & inference wrapper
 ├── drqn_model.py    # DRQN architecture
 └── model.pth        # Trained model weights

tests/
 ├── test_api.py
 └── test_model.py

Dockerfile
requirements.txt
.github/workflows/ci.yml
🐳 Run Locally (Docker)

Build the container:

docker build -t ddos-api .

Run the service:

docker run -p 8000:8000 ddos-api

Open:

http://localhost:8000/docs
🔁 CI/CD

Every push to main:

Runs automated tests

Builds Docker image

Tags image with commit SHA

Pushes image to DockerHub

Example image tags:

ddos-api:latest
ddos-api:3f2a8c1
📊 Model Inputs

The API expects a JSON payload:

{
  "observation": [request_rate, error_rate, latency, queue_length, cpu_usage]
}

Returns:

{
  "mitigation_action": 2
}
📈 Key Results

Stable reward convergence

Bounded response latency under attack

Improved F1 score over training

Progressive mitigation behavior

Cost-aware action selection

⚠️ Limitations

Simulated environment

Single-server setting

Manual reward tuning

🔮 Future Work

Multi-server deployment

Real traffic dataset integration

Multi-agent reinforcement learning

Automated model retraining

Production monitoring integration

👨‍💻 Author

Your Name
B.Tech Information Technology
Deep Reinforcement Learning | ML Deployment | DevOps
