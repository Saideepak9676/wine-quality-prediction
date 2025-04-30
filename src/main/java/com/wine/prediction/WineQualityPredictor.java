package com.wine.prediction;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.PipelineModel;
import org.apache.hadoop.fs.s3a.S3AFileSystem;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.StructType;

public class WineQualityPredictor {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: WineQualityPredictor <testDataPath>");
            System.exit(1);
        }

        String testDataPath = args[0];
        String modelPath = "s3a://wine-prediction-quality-check/wine-quality-model";

        try {
            // Create Spark session
            SparkSession spark = SparkSession.builder()
                    .appName("WineQualityPredictor")
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
            
            // Load the saved model
            System.out.println("Loading model from: " + modelPath);
            PipelineModel model = PipelineModel.load(modelPath);

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

            // Read the test data
            Dataset<Row> testData = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .schema(schema)
                .csv(testDataPath);

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

            Dataset<Row> testDataWithFeatures = assembler.transform(testData);

            // Make predictions
            Dataset<Row> predictions = model.transform(testDataWithFeatures);

            // Show predictions
            predictions.select("features", "prediction").show(5);

            // Calculate and display metrics
            predictions = predictions.withColumn("\"\"\"quality\"\"\"", 
                predictions.col("\"\"\"quality\"\"\"").cast(DataTypes.DoubleType));
            
            double mse = predictions.select(functions.pow(
                functions.col("\"\"\"quality\"\"\"").minus(functions.col("prediction")), 2))
                .agg(functions.avg("POWER((\"\"\"quality\"\"\" - prediction), 2)"))
                .first().getDouble(0);
            
            double rmse = Math.sqrt(mse);
            System.out.println("Root Mean Squared Error (RMSE) = " + rmse);

            // Stop Spark session
            spark.stop();
        } catch (Exception e) {
            System.err.println("Error occurred: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
} 