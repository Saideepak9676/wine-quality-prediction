package model;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class DataProcessor {
    private final SparkSession spark;

    public DataProcessor(SparkSession spark) {
        this.spark = spark;
    }

    public Dataset<Row> loadAndProcessData(String s3Path) {
        // Define schema for the wine dataset
        StructType schema = new StructType(new StructField[]{
                DataTypes.createStructField("fixed_acidity", DataTypes.DoubleType, true),
                DataTypes.createStructField("volatile_acidity", DataTypes.DoubleType, true),
                DataTypes.createStructField("citric_acid", DataTypes.DoubleType, true),
                DataTypes.createStructField("residual_sugar", DataTypes.DoubleType, true),
                DataTypes.createStructField("chlorides", DataTypes.DoubleType, true),
                DataTypes.createStructField("free_sulfur_dioxide", DataTypes.DoubleType, true),
                DataTypes.createStructField("total_sulfur_dioxide", DataTypes.DoubleType, true),
                DataTypes.createStructField("density", DataTypes.DoubleType, true),
                DataTypes.createStructField("pH", DataTypes.DoubleType, true),
                DataTypes.createStructField("sulphates", DataTypes.DoubleType, true),
                DataTypes.createStructField("alcohol", DataTypes.DoubleType, true),
                DataTypes.createStructField("quality", DataTypes.DoubleType, true)
        });

        // Load data from S3
        System.out.println("Loading training data from: " + s3Path);
        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .schema(schema)
                .csv(s3Path);

        // Handle missing values
        data = data.na().fill(0.0);

        return data;
    }
} 