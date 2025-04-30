FROM maven:3.8.4-openjdk-8

# Add metadata
LABEL maintainer="saideepak9676"
LABEL version="1.0"
LABEL description="Wine Quality Prediction Model using Spark MLlib"

# Install required packages
RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Spark
RUN wget https://archive.apache.org/dist/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz && \
    tar xvf spark-3.3.1-bin-hadoop3.tgz --no-same-owner && \
    mv spark-3.3.1-bin-hadoop3 /opt/spark && \
    rm spark-3.3.1-bin-hadoop3.tgz

# Set environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Create app directory
WORKDIR /app

# Copy the Maven project files
COPY pom.xml .
COPY src ./src
COPY dataset ./dataset

# Build the application
RUN mvn clean package

# Create a directory for AWS credentials
RUN mkdir -p /root/.aws

# Create a script to run the application
COPY convert_to_word.sh /app/
RUN chmod +x /app/convert_to_word.sh

# Set the entry point
ENTRYPOINT ["/app/convert_to_word.sh"] 