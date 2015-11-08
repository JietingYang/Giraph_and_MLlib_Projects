import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.tree.RandomForest;
import scala.Tuple2;

import java.util.HashMap;
import java.util.regex.Pattern;

public final class RandomForestMP {
	private static class DataToPoint implements Function<String, LabeledPoint> {
		private static final Pattern SPACE = Pattern.compile(",");

		public LabeledPoint call(String line) throws Exception {
			String[] tok = SPACE.split(line);
			double label = Double.parseDouble(tok[tok.length - 1]);
			double[] point = new double[tok.length - 1];
			for (int i = 0; i < tok.length - 1; ++i) {
				point[i] = Double.parseDouble(tok[i]);
			}
			return new LabeledPoint(label, Vectors.dense(point));
		}
	}
    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println(
                    "Usage: RandomForestMP <training_data> <test_data> <results>");
            System.exit(1);
        }
        String training_data_path = args[0];
        String test_data_path = args[1];
        String results_path = args[2];

        SparkConf sparkConf = new SparkConf().setAppName("RandomForestMP");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        final NaiveBayesModel model;

        Integer numClasses = 2;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        Integer numTrees = 3;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;
        Integer seed = 12345;

		// TODO
		JavaRDD<LabeledPoint> train = sc.textFile(training_data_path).map(new DataToPoint());
		JavaRDD<LabeledPoint> test = sc.textFile(test_data_path).map(new DataToPoint());
		model = NaiveBayes.train(train.rdd(), 1.0);
		JavaPairRDD<Double, Double> predictionAndLabel = test
				.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					public Tuple2<Double, Double> call(LabeledPoint p) {
						return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
					}
				});
		double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
			public Boolean call(Tuple2<Double, Double> pl) {
				return pl._1().equals(pl._2());
			}
		}).count() / (double) test.count();

        JavaRDD<LabeledPoint> results = test.map(new Function<LabeledPoint, LabeledPoint>() {
//            public LabeledPoint call(Vector points) {
//                return new LabeledPoint(model.predict(points), points);
//            }

			public LabeledPoint call(LabeledPoint arg0) throws Exception {
				return new LabeledPoint(model.predict((Vector) arg0), (Vector) arg0);
			}
        });

        results.saveAsTextFile(results_path);
        System.out.println(accuracy);
        sc.stop();
    }

}
