# Wine Quality Prediction Model - CS 643 Cloud Computing Assignment

This project implements a wine quality prediction model using Apache Spark and MLlib, trained in parallel on AWS EC2 instances. The model is containerized using Docker for easy deployment.

## Project Overview

The project consists of two main components:
1. Parallel model training on 4 EC2 instances using Spark MLlib
2. Single-machine prediction application (with Docker container support)

## Prerequisites

- AWS Account with appropriate permissions
- Java 8 or higher
- Maven 3.8.4 or higher
- Docker (for containerized deployment)
- Git

## Dataset Structure

The project uses three datasets:
- `TrainingDataset.csv`: Used for training the model
- `ValidationDataset.csv`: Used for model validation and parameter tuning
- `TestDataset.csv`: Used for testing the prediction application

## Setup Instructions

### 1. AWS Environment Setup

1. Launch 4 EC2 instances (t2.micro or larger) with Ubuntu Linux
2. Configure security groups:
   - Allow SSH (port 22)
   - Allow Spark communication (ports 7077, 8080)
   - Allow inter-instance communication
3. Set up IAM roles with S3 permissions:
   - `s3:GetObject`
   - `s3:PutObject`
   - `s3:ListBucket`
4. Install required software on all instances:
   ```bash
   sudo apt-get update
   sudo apt-get install -y openjdk-8-jdk maven
   ```

### 2. Project Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/saisrinivas194/CS_643-wine-quality-prediction
   cd wine-prediction
   ```

2. Build the application:
   ```bash
   mvn clean package
   ```

### 3. Running the Model Training

1. Copy the JAR file to all 4 EC2 instances
2. Start Spark master on the master node:
   ```bash
   /usr/local/spark/sbin/start-master.sh
   ```
3. Start Spark workers on all nodes:
   ```bash
   /usr/local/spark/sbin/start-worker.sh spark://<master-ip>:7077
   ```
4. Run the training application:
   ```bash
   spark-submit --master spark://<master-ip>:7077 \
     --class com.example.WineModel \
     target/wine-prediction-1.0-SNAPSHOT.jar
   ```

### 4. Running the Prediction Application

#### Without Docker:

1. Copy the JAR file to a single EC2 instance
2. Run the prediction application:
   ```bash
   java -jar target/wine-prediction-1.0-SNAPSHOT.jar
   ```

#### With Docker:

1. Build the Docker image:
   ```bash
   docker build -t wine-prediction .
   ```

2. Run the container:
   ```bash
   docker run -v /path/to/test/data:/data wine-prediction
   ```

## Model Performance

The model achieves an RMSE of 0.5515 on the validation dataset, indicating good prediction accuracy.

## Docker Hub

The Docker image is available at: [saideepak9676/wine-quality-prediction](https://hub.docker.com/r/saideepak9676/wine-quality-prediction)

You can pull the image using:
```bash
docker pull saideepak9676/wine-quality-prediction:latest
```

To run the pulled image:
```bash
docker run -d \
  -e AWS_ACCESS_KEY_ID=your_access_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret_key \
  -e AWS_DEFAULT_REGION=your_region \
  saideepak9676/wine-quality-prediction:latest
```

## Project Structure

- `src/main/java/com/example/WineModel.java`: Main application for model training
- `src/main/java/com/example/WinePrediction.java`: Prediction application
- `Dockerfile`: Container configuration
- `pom.xml`: Maven project configuration

## Implementation Details

### Model Training
- Uses Spark MLlib for parallel processing
- Implements cross-validation for model tuning
- Saves trained model to S3 bucket
- Achieved RMSE of 0.5515 on validation dataset
- Model parameters:
  - Linear Regression with regularization
  - Cross-validation folds: 5
  - Feature scaling: StandardScaler

### Prediction Application
- Loads model from S3
- Processes input data
- Outputs prediction results
- Supports both CSV and JSON input formats
- Includes error handling and validation

### Docker Container
- Based on Maven 3.8.4 with OpenJDK 8
- Includes all necessary dependencies
- Mounts data directory for input/output
- Environment variables for configuration:
  - AWS_REGION
  - S3_BUCKET_NAME
  - MODEL_PATH

## Actual Implementation Results

### Model Performance
- Training RMSE: 0.5515
- Validation RMSE: 0.5515
- Model saved to: s3://your-bucket-name/models/wine-model

### AWS Configuration
- Instance type: t2.micro
- Region: us-east-1
- Security groups configured for:
  - Spark communication (7077, 8080)
  - SSH access (22)
  - Inter-instance communication

### Docker Container
- Image: saisrinivas194/wine-prediction
- Size: ~500MB
- Build time: ~5 minutes
- Memory requirements: 2GB minimum

## Step-by-Step Implementation Process

1. **AWS Setup**
   - Launched 4 EC2 instances
   - Configured security groups
   - Set up IAM roles
   - Installed required software

2. **Spark Cluster Setup**
   - Started Spark master
   - Configured worker nodes
   - Verified cluster connectivity

3. **Model Training**
   - Uploaded training data to S3
   - Ran parallel training
   - Validated model performance
   - Saved model to S3

4. **Prediction Application**
   - Developed standalone application
   - Implemented model loading
   - Added input validation
   - Tested with validation dataset

5. **Docker Container**
   - Created Dockerfile
   - Built container
   - Tested locally
   - Pushed to Docker Hub

## Testing and Validation

1. **Model Testing**
   - Used ValidationDataset.csv
   - Achieved RMSE: 0.5515
   - Verified prediction accuracy

2. **Application Testing**
   - Tested with various input formats
   - Verified error handling
   - Confirmed S3 integration

3. **Docker Testing**
   - Built and ran container
   - Verified data mounting
   - Tested environment variables

## Docker Implementation

### Prerequisites
- Docker installed on your system
- Docker Compose installed
- AWS credentials configured

### Setup Instructions

1. Copy the environment template and configure your AWS credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials and S3 configuration
   ```

2. Build and run using Docker Compose:
   ```bash
   docker-compose up --build
   ```

3. To run in detached mode:
   ```bash
   docker-compose up -d
   ```

4. To view logs:
   ```bash
   docker-compose logs -f
   ```

5. To stop the container:
   ```bash
   docker-compose down
   ```

### Docker Configuration Details

The Docker implementation includes:
- Maven 3.8.4 with OpenJDK 8
- Spark 3.3.1 with Hadoop 3
- Python 3 for PySpark support
- AWS SDK for S3 integration
- Volume mounts for:
  - Dataset directory
  - Source code
  - Target directory
  - AWS credentials

### Environment Variables

The following environment variables are required:
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_DEFAULT_REGION`: Your AWS region
- `S3_BUCKET`: S3 bucket name for model storage
- `S3_MODEL_PATH`: Path within the S3 bucket for model storage

