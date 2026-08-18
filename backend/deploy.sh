#!/bin/bash
# deploy.sh
# Run this on your fresh Ubuntu EC2 instance

set -e

echo "Updating packages..."
sudo apt update && sudo apt upgrade -y

echo "Installing Python 3.10 and pip..."
sudo apt install python3-pip python3-venv git -y

echo "Setting up backend..."
cd ~/
# Assuming repo is cloned to ~/RAG
cd RAG/backend

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setting up systemd service for uvicorn..."
cat << 'EOF' | sudo tee /etc/systemd/system/rag-backend.service
[Unit]
Description=RAG Backend Service
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/RAG/backend
Environment="PATH=/home/ubuntu/RAG/backend/venv/bin"
ExecStart=/home/ubuntu/RAG/backend/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rag-backend
sudo systemctl start rag-backend

echo "Deployment complete! Uvicorn is running on port 8000."
echo "Check status with: sudo systemctl status rag-backend"
