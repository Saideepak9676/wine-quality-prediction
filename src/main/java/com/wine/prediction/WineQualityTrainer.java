package com.wine.prediction;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.hadoop.fs.s3a.S3AFileSystem;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.sql.types.StructType;
import java.io.IOException;

public class WineQualityTrainer {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: WineQualityTrainer <trainingDataPath>");
            System.exit(1);
        }

        String trainingDataPath = args[0];
        String modelPath = "s3a://wine-prediction-quality-check/wine-quality-model";

        try {
            // Create Spark session
            SparkSession spark = SparkSession.builder()
                    .appName("WineQualityTrainer")
                    .getOrCreate();

            // Configure AWS credentials
            String accessKey = System.getenv("AWS_ACCESS_KEY_ID");
            String secretKey = System.getenv("AWS_SECRET_ACCESS_KEY");
            String sessionToken = System.getenv("AWS_SESSION_TOKEN");
            String region = System.getenv("AWS_REGION");
            
            if (accessKey == null || secretKey == null || sessionToken == null) {
                throw new RuntimeException("AWS credentials not found in environment variables");
            }

            // Configure Hadoop for S3 access
            spark.sparkContext().hadoopConfiguration().set("fs.s3a.access.key", accessKey);
            spark.sparkContext().hadoopConfiguration().set("fs.s3a.secret.key", secretKey);
            spark.sparkContext().hadoopConfiguration().set("fs.s3a.session.token", sessionToken);
            spark.sparkContext().hadoopConfiguration().set("fs.s3a.endpoint", "s3." + region + ".amazonaws.com");
            spark.sparkContext().hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem");
            spark.sparkContext().hadoopConfiguration().set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider");
            spark.sparkContext().hadoopConfiguration().set("fs.s3a.connection.ssl.enabled", "true");
            spark.sparkContext().hadoopConfiguration().set("fs.s3a.path.style.access", "false");
            spark.sparkContext().hadoopConfiguration().set("fs.s3a.region", region);
            
            // Define the schema
            StructType schema = new StructType()
                .add("\"\"\"fixed acidity\"\"\"", "double")
                .add("\"\"\"volatile acidity\"\"\"", "double")
                .add("\"\"\"citric acid\"\"\"", "double")
                .add("\"\"\"residual sugar\"\"\"", "double")
                .add("\"\"\"chlorides\"\"\"", "double")
                .add("\"\"\"free sulfur dioxide\"\"\"", "double")
                .add("\"\"\"total sulfur dioxide\"\"\"", "double")
                .add("\"\"\"density\"\"\"", "double")
                .add("\"\"\"pH\"\"\"", "double")
                .add("\"\"\"sulphates\"\"\"", "double")
                .add("\"\"\"alcohol\"\"\"", "double")
                .add("\"\"\"quality\"\"\"", "double");

            // Read the training data
            System.out.println("Loading training data from: " + trainingDataPath);
            Dataset<Row> trainingData = spark.read()
                    .option("header", "true")
                    .option("delimiter", ";")
                    .schema(schema)
                    .csv(trainingDataPath);

            // Define the feature columns
            String[] featureColumns = {
                "\"\"\"fixed acidity\"\"\"",
                "\"\"\"volatile acidity\"\"\"",
                "\"\"\"citric acid\"\"\"",
                "\"\"\"residual sugar\"\"\"",
                "\"\"\"chlorides\"\"\"",
                "\"\"\"free sulfur dioxide\"\"\"",
                "\"\"\"total sulfur dioxide\"\"\"",
                "\"\"\"density\"\"\"",
                "\"\"\"pH\"\"\"",
                "\"\"\"sulphates\"\"\"",
                "\"\"\"alcohol\"\"\""
            };

            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureColumns)
                    .setOutputCol("features");

            // Create Random Forest model
            RandomForestRegressor rf = new RandomForestRegressor()
                    .setLabelCol("\"\"\"quality\"\"\"")
                    .setFeaturesCol("features")
                    .setNumTrees(100)
                    .setMaxDepth(10);

            // Create pipeline
            Pipeline pipeline = new Pipeline()
                    .setStages(new PipelineStage[]{assembler, rf});

            // Train model
            System.out.println("Training model...");
            PipelineModel model = pipeline.fit(trainingData);

            // Save model
            System.out.println("Saving model to: " + modelPath);
            try {
                model.write().overwrite().save(modelPath);
            } catch (IOException e) {
                System.err.println("Error saving model: " + e.getMessage());
                e.printStackTrace();
                System.exit(1);
            }

            // Stop Spark session
            spark.stop();
        } catch (Exception e) {
            System.err.println("Error occurred: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
} 