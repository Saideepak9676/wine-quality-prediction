package com.mlearning.spark;

import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class PredictWineQuality {

    // ✅ You can rename this to your full name or student ID for clarity
    public static final Logger logger = LogManager.getLogger(PredictWineQuality.class);

    // ✅ Replace 'dataset/' with your own S3 bucket prefix if different
    private static final String BUCKET_NAME = System.getProperty("BUCKET_NAME", "dataset/");

    // ✅ These should be set when you run the app using `-DACCESS_KEY_ID=...`
    private static final String ACCESS_KEY_ID = System.getProperty("ACCESS_KEY_ID");
    private static final String SECRET_KEY = System.getProperty("SECRET_KEY");

    // ✅ Replace filenames if you're using different test file or model folder
    private static final String TESTING_DATASET = BUCKET_NAME + "TestDataset.csv";
    private static final String MODEL_PATH = BUCKET_NAME + "LogisticRegression";

    private static final String MASTER_URI = "local[*]";

    public static void main(String[] args) {

        // Optional: silence Spark logs for better readability
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);
        Logger.getLogger("breeze.optimize").setLevel(Level.ERROR);
        Logger.getLogger("com.amazonaws.auth").setLevel(Level.DEBUG);

        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction") // ✅ Replace app name with assignment-specific title
                .master(MASTER_URI)
                .config("spark.executor.memory", "3g")
                .config("spark.driver.memory", "3g")
                .getOrCreate();

        // Set AWS S3 credentials
        if (StringUtils.isNotEmpty(ACCESS_KEY_ID) && StringUtils.isNotEmpty(SECRET_KEY)) {
            spark.sparkContext().hadoopConfiguration().set("fs.s3a.access.key", ACCESS_KEY_ID);
            spark.sparkContext().hadoopConfiguration().set("fs.s3a.secret.key", SECRET_KEY);
        }

        // Run prediction
        PredictWineQuality predictor = new PredictWineQuality();
        predictor.runPrediction(spark);
    }

    public void runPrediction(SparkSession spark) {
        System.out.println("Evaluating model with test dataset...\n");

        // Load trained model from S3
        PipelineModel model = PipelineModel.load(MODEL_PATH);

        // Load test data
        Dataset<Row> testData = getDataFrame(spark, true, TESTING_DATASET).cache();

        // Predict
        Dataset<Row> predictions = model.transform(testData).cache();
        predictions.select("features", "label", "prediction").show(5, false);

        // Evaluate predictions
        printMetrics(predictions);
    }

    public Dataset<Row> getDataFrame(SparkSession spark, boolean transform, String path) {

        Dataset<Row> data = spark.read().format("csv")
                .option("header", "true")
                .option("sep", ";")
                .option("quote", "\"")
                .option("inferSchema", true)
                .load(path);

        Dataset<Row> cleaned = data
                .withColumnRenamed("quality", "label")
                .select("label", "alcohol", "sulphates", "pH", "density",
                        "free sulfur dioxide", "total sulfur dioxide",
                        "chlorides", "residual sugar", "citric acid",
                        "volatile acidity", "fixed acidity")
                .na().drop();

        if (transform) {
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(new String[]{
                            "alcohol", "sulphates", "pH", "density",
                            "free sulfur dioxide", "total sulfur dioxide",
                            "chlorides", "residual sugar", "citric acid",
                            "volatile acidity", "fixed acidity"
                    })
                    .setOutputCol("features");

            cleaned = assembler.transform(cleaned).select("label", "features");
        }

        return cleaned;
    }

    public void printMetrics(Dataset<Row> predictions) {
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();

        evaluator.setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Model Accuracy: " + accuracy);
        System.out.println("Test Error: " + (1.0 - accuracy));

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);
        System.out.println("F1 Score: " + f1);

        evaluator.setMetricName("weightedPrecision");
        double precision = evaluator.evaluate(predictions);
        System.out.println("Weighted Precision: " + precision);

        evaluator.setMetricName("weightedRecall");
        double recall = evaluator.evaluate(predictions);
        System.out.println("Weighted Recall: " + recall);
    }
}
