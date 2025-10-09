"""
Meng AI Deployment Script
========================

Deployment script for Meng AI with:
1. Model training and preparation
2. Docker containerization
3. Production deployment
4. Health checks and monitoring
5. Scaling and load balancing

Author: Meng AI Development Team
"""

import os
import sys
import subprocess
import json
import time
import requests
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MengAIDeployer:
    """Deployment manager for Meng AI"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        
        # Create directories
        for dir_path in [self.models_dir, self.data_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load deployment configuration"""
        default_config = {
            'model': {
                'size': 'medium',
                'train_data_path': None,
                'epochs': 5,
                'batch_size': 16,
                'learning_rate': 1e-4
            },
            'deployment': {
                'docker_image': 'meng-ai:latest',
                'container_name': 'meng-ai-server',
                'port': 8000,
                'replicas': 1,
                'memory_limit': '8G',
                'cpu_limit': '4'
            },
            'monitoring': {
                'enable_prometheus': True,
                'enable_grafana': True,
                'health_check_interval': 30
            },
            'scaling': {
                'min_replicas': 1,
                'max_replicas': 5,
                'target_cpu': 70,
                'target_memory': 80
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                # Merge with defaults
                for key, value in file_config.items():
                    if key in default_config:
                        default_config[key].update(value)
        
        return default_config
    
    def train_model(self, data_path: Optional[str] = None) -> bool:
        """Train the Meng AI model"""
        logger.info("Starting model training...")
        
        try:
            # Import training modules
            from meng_ai_trainer import MengAITrainer, TrainingConfig, ModelSize
            from simple_tokenizer import create_sample_dataset
            
            # Prepare data
            if data_path and os.path.exists(data_path):
                logger.info(f"Loading data from {data_path}")
                with open(data_path, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]
            else:
                logger.info("Using sample dataset")
                texts = create_sample_dataset()
            
            logger.info(f"Training on {len(texts)} texts")
            
            # Training configuration
            config = TrainingConfig(
                model_size=ModelSize(self.config['model']['size']),
                vocab_size=10000,
                batch_size=self.config['model']['batch_size'],
                max_epochs=self.config['model']['epochs'],
                learning_rate=self.config['model']['learning_rate'],
                use_amp=True,
                eval_steps=100,
                save_steps=200
            )
            
            # Initialize trainer
            trainer = MengAITrainer(config)
            
            # Train model
            output_dir = self.models_dir / "trained_model"
            trainer.train(texts, str(output_dir))
            
            logger.info(f"Model training completed. Saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def build_docker_image(self) -> bool:
        """Build Docker image"""
        logger.info("Building Docker image...")
        
        try:
            # Build Docker image
            cmd = [
                'docker', 'build',
                '-t', self.config['deployment']['docker_image'],
                '-f', 'Dockerfile',
                '.'
            ]
            
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Docker build failed: {result.stderr}")
                return False
            
            logger.info("Docker image built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Docker build error: {e}")
            return False
    
    def deploy_container(self) -> bool:
        """Deploy Docker container"""
        logger.info("Deploying Docker container...")
        
        try:
            # Stop existing container if running
            self._stop_container()
            
            # Run new container
            cmd = [
                'docker', 'run',
                '-d',
                '--name', self.config['deployment']['container_name'],
                '-p', f"{self.config['deployment']['port']}:8000",
                '--memory', self.config['deployment']['memory_limit'],
                '--cpus', self.config['deployment']['cpu_limit'],
                '--restart', 'unless-stopped',
                '-v', f"{self.models_dir}:/app/models",
                '-v', f"{self.data_dir}:/app/data",
                '-v', f"{self.logs_dir}:/app/logs",
                self.config['deployment']['docker_image']
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Container deployment failed: {result.stderr}")
                return False
            
            logger.info("Container deployed successfully")
            
            # Wait for container to start
            time.sleep(10)
            
            # Check health
            if self._check_health():
                logger.info("Container is healthy and ready")
                return True
            else:
                logger.error("Container health check failed")
                return False
                
        except Exception as e:
            logger.error(f"Container deployment error: {e}")
            return False
    
    def _stop_container(self):
        """Stop existing container"""
        try:
            subprocess.run(['docker', 'stop', self.config['deployment']['container_name']], 
                         capture_output=True)
            subprocess.run(['docker', 'rm', self.config['deployment']['container_name']], 
                         capture_output=True)
        except:
            pass
    
    def _check_health(self) -> bool:
        """Check container health"""
        try:
            url = f"http://localhost:{self.config['deployment']['port']}/health"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def deploy_with_compose(self) -> bool:
        """Deploy using Docker Compose"""
        logger.info("Deploying with Docker Compose...")
        
        try:
            # Stop existing services
            subprocess.run(['docker-compose', 'down'], cwd=self.project_root, capture_output=True)
            
            # Start services
            result = subprocess.run(['docker-compose', 'up', '-d'], 
                                  cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Docker Compose deployment failed: {result.stderr}")
                return False
            
            logger.info("Docker Compose deployment successful")
            
            # Wait for services to start
            time.sleep(30)
            
            # Check health
            if self._check_health():
                logger.info("Services are healthy and ready")
                return True
            else:
                logger.error("Service health check failed")
                return False
                
        except Exception as e:
            logger.error(f"Docker Compose deployment error: {e}")
            return False
    
    def scale_service(self, replicas: int) -> bool:
        """Scale the service"""
        logger.info(f"Scaling service to {replicas} replicas...")
        
        try:
            cmd = ['docker-compose', 'up', '-d', '--scale', 
                   f"meng-ai-server={replicas}"]
            
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Scaling failed: {result.stderr}")
                return False
            
            logger.info(f"Service scaled to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Scaling error: {e}")
            return False
    
    def monitor_service(self) -> Dict[str, Any]:
        """Monitor service status"""
        try:
            # Get container stats
            cmd = ['docker', 'stats', '--no-stream', '--format', 
                   'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}']
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Get health status
            health_status = self._check_health()
            
            # Get service info
            url = f"http://localhost:{self.config['deployment']['port']}/metrics"
            try:
                response = requests.get(url, timeout=5)
                metrics = response.json() if response.status_code == 200 else {}
            except:
                metrics = {}
            
            return {
                'health': health_status,
                'stats': result.stdout if result.returncode == 0 else "N/A",
                'metrics': metrics,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            return {'health': False, 'error': str(e)}
    
    def create_config_file(self, config_path: str = "deployment_config.yaml"):
        """Create deployment configuration file"""
        config = {
            'model': {
                'size': 'medium',  # small, medium, large, xl
                'train_data_path': None,  # Path to training data
                'epochs': 5,
                'batch_size': 16,
                'learning_rate': 1e-4
            },
            'deployment': {
                'docker_image': 'meng-ai:latest',
                'container_name': 'meng-ai-server',
                'port': 8000,
                'replicas': 1,
                'memory_limit': '8G',
                'cpu_limit': '4'
            },
            'monitoring': {
                'enable_prometheus': True,
                'enable_grafana': True,
                'health_check_interval': 30
            },
            'scaling': {
                'min_replicas': 1,
                'max_replicas': 5,
                'target_cpu': 70,
                'target_memory': 80
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Deployment configuration created: {config_path}")
    
    def full_deployment(self) -> bool:
        """Perform full deployment"""
        logger.info("Starting full deployment...")
        
        # 1. Train model
        if not self.train_model():
            logger.error("Model training failed")
            return False
        
        # 2. Build Docker image
        if not self.build_docker_image():
            logger.error("Docker image build failed")
            return False
        
        # 3. Deploy with Docker Compose
        if not self.deploy_with_compose():
            logger.error("Docker Compose deployment failed")
            return False
        
        logger.info("Full deployment completed successfully!")
        return True

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Meng AI Deployment")
    parser.add_argument("--config", type=str, help="Deployment configuration file")
    parser.add_argument("--create-config", action="store_true", help="Create sample configuration")
    parser.add_argument("--train", action="store_true", help="Train model only")
    parser.add_argument("--build", action="store_true", help="Build Docker image only")
    parser.add_argument("--deploy", action="store_true", help="Deploy container only")
    parser.add_argument("--compose", action="store_true", help="Deploy with Docker Compose")
    parser.add_argument("--scale", type=int, help="Scale service to N replicas")
    parser.add_argument("--monitor", action="store_true", help="Monitor service status")
    parser.add_argument("--full", action="store_true", help="Full deployment")
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = MengAIDeployer(args.config)
    
    if args.create_config:
        deployer.create_config_file()
        return
    
    if args.train:
        success = deployer.train_model()
        sys.exit(0 if success else 1)
    
    if args.build:
        success = deployer.build_docker_image()
        sys.exit(0 if success else 1)
    
    if args.deploy:
        success = deployer.deploy_container()
        sys.exit(0 if success else 1)
    
    if args.compose:
        success = deployer.deploy_with_compose()
        sys.exit(0 if success else 1)
    
    if args.scale:
        success = deployer.scale_service(args.scale)
        sys.exit(0 if success else 1)
    
    if args.monitor:
        status = deployer.monitor_service()
        print(json.dumps(status, indent=2))
        return
    
    if args.full:
        success = deployer.full_deployment()
        sys.exit(0 if success else 1)
    
    # Default: show help
    parser.print_help()

if __name__ == "__main__":
    main()
