package model;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;

public class WineModel {
    public static void main(String[] args) {
        // Create Spark session with S3 configuration
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction")
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
                .config("spark.hadoop.fs.s3a.impl.disable.cache", "true")
                .config("spark.hadoop.fs.s3a.connection.maximum", "1000")
                .config("spark.hadoop.fs.s3a.connection.timeout", "50000")
                .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000")
                .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
                .getOrCreate();

        try {
            // Print Spark configuration for debugging
            System.out.println("Spark Configuration:");
            System.out.println("S3A Implementation: " + spark.conf().get("spark.hadoop.fs.s3a.impl"));
            System.out.println("S3A Endpoint: " + spark.conf().get("spark.hadoop.fs.s3a.endpoint"));
            System.out.println("S3A Credentials Provider: " + spark.conf().get("spark.hadoop.fs.s3a.aws.credentials.provider"));

            // Load and process data
            System.out.println("Starting data processing...");
            DataProcessor dataProcessor = new DataProcessor(spark);
            String s3Path = "s3a://wine-quality-ml-bucket/TrainingDataset.csv";
            Dataset<Row> processedData = dataProcessor.loadAndProcessData(s3Path);

            // Split data into training and testing sets
            Dataset<Row>[] splits = processedData.randomSplit(new double[]{0.8, 0.2}, 42L);
            Dataset<Row> trainingData = splits[0];
            Dataset<Row> testData = splits[1];

            // Create feature vector
            String[] featureCols = {"fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH",
                    "sulphates", "alcohol"};
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureCols)
                    .setOutputCol("features");

            // Scale features
            StandardScaler scaler = new StandardScaler()
                    .setInputCol("features")
                    .setOutputCol("scaledFeatures")
                    .setWithStd(true)
                    .setWithMean(true);

            // Create Random Forest model
            RandomForestRegressor rf = new RandomForestRegressor()
                    .setLabelCol("quality")
                    .setFeaturesCol("scaledFeatures")
                    .setNumTrees(100)
                    .setMaxDepth(10);

            // Create pipeline
            Pipeline pipeline = new Pipeline()
                    .setStages(new PipelineStage[]{assembler, scaler, rf});

            // Train model
            System.out.println("Training model...");
            PipelineModel model = pipeline.fit(trainingData);

            // Make predictions
            Dataset<Row> predictions = model.transform(testData);

            // Evaluate model
            RegressionEvaluator evaluator = new RegressionEvaluator()
                    .setLabelCol("quality")
                    .setPredictionCol("prediction")
                    .setMetricName("rmse");

            double rmse = evaluator.evaluate(predictions);
            System.out.println("Root Mean Square Error = " + rmse);

            // Save model to S3
            System.out.println("Saving model to S3...");
            model.write().overwrite().save("s3a://wine-quality-ml-bucket/models/wine_quality_model");

            System.out.println("Model training and saving completed successfully!");

        } catch (Exception e) {
            System.err.println("Error in main process: " + e.getMessage());
            e.printStackTrace();
            // Print more detailed error information
            if (e.getCause() != null) {
                System.err.println("Caused by: " + e.getCause().getMessage());
                e.getCause().printStackTrace();
            }
        } finally {
            spark.stop();
        }
    }
} 