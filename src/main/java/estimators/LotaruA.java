package estimators;

import jep.Interpreter;
import jep.JepException;
import jep.SharedInterpreter;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.math.NumberUtils;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.javatuples.Septet;
import org.javatuples.Sextet;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class LotaruA implements Estimator {
    Interpreter interp; // jep requires a python environment with the python equivalent installed
    // pip install jep
    // additionally, make sure sklearn and numpy are available to python

    public LotaruA() {
        interp = new SharedInterpreter();
        try { // do imports only once
            interp.exec("from sklearn.metrics import r2_score");
            interp.exec("from sklearn.linear_model import BayesianRidge");
            interp.exec("from sklearn.linear_model import LinearRegression");
            interp.exec("import numpy");

            interp.exec("numpy.set_printoptions(threshold=numpy.inf)");
            interp.exec("numpy.set_printoptions(linewidth=numpy.inf)");
        } catch (JepException e) {
            throw new RuntimeException("Python imports failed. Cant run LotaruA", e);
        }
    }

    public Septet<String, String, String, double[], double[], double[], double[]> estimateWith1DInput(String taskname, String resourceToPredict, double[] ids, double[] train_x, double[] train_y, double[] test_x, double[] test_y, double factor) {

        // Extra Parameter nicht train_X zurück geben
        if (train_x.length != train_y.length) {
            throw new RuntimeException("Length of X should be equal to length Y");
        }

        var pearson = calculatePearson(train_x, train_y);


        if (pearson < 0.75 || Double.isNaN(pearson)) {
            DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics(train_y);

            double median_predicted = descriptiveStatistics.getPercentile(50);


            double[] toReturnError = new double[test_y.length];

            for (int i = 0; i < toReturnError.length; i++) {
                toReturnError[i] = Math.abs((median_predicted - test_y[i]) / test_y[i]);
            }

            double[] median_predicted_arr = new double[test_y.length];

            for (int i = 0; i < test_y.length; i++) {
                median_predicted_arr[i] = median_predicted;
            }


            return new Septet<>(taskname, "Lotaru-A", resourceToPredict, ids, median_predicted_arr, test_y, toReturnError);
        }

        try {
            interp.set("input_trX", train_x);
            interp.exec("train_X = numpy.array(input_trX, dtype=float)");
            interp.set("input_trY", train_y);
            interp.exec("train_Y = numpy.array(input_trY, dtype=float)");
            interp.set("input_teX", test_x);
            interp.exec("test_X = numpy.array(input_teX, dtype=float)");
            interp.set("input_trY", test_y);
            interp.exec("test_Y = numpy.array(input_trY, dtype=float)");
        } catch (JepException e) {
            throw new RuntimeException("Failed to transfer input data to python bayes", e);
        }

        try {
            // Creating and training model
            interp.exec("model = BayesianRidge(fit_intercept=True)");
            interp.exec("model.fit(train_X.reshape(-1, 1), train_Y)");
        } catch (JepException e) {
            throw new RuntimeException("Failed to fit python bayes", e);
        }

        try {
            // Model making a prediction on test data
            interp.exec("prediction = model.predict(test_X.reshape(-1, 1))");
        } catch (JepException e) {
            throw new RuntimeException("Couldn't get bayes prediction", e);
        }

        try {
            ArrayList<Double> tmp = interp.getValue("list(prediction)", ArrayList.class);
            double[] predicted = tmp.stream().map(x -> x * factor).mapToDouble(Double::valueOf).toArray();

            double[] toReturnError = new double[test_y.length];
            for (int i = 0; i < predicted.length; i++) {
                toReturnError[i] = Math.abs((predicted[i] - test_y[i]) / test_y[i]);
                if (toReturnError[i] > 1) {
                    System.out.println(i);
                }
            }

            return new Septet<>(taskname, "Lotaru-A", resourceToPredict, ids, predicted, test_y, toReturnError);
        } catch (JepException e) {
            throw new RuntimeException("Couldn't get results from python bayes", e);
        }
    }


    public Sextet<String, String, String, Double, Double, Double> estimateWith2DInput(String taskname, String resourceToPredict, double[][] train_x, double[] train_y, double[][] test_x, double[] test_y, double factor) {


        if (train_x[0].length != test_x[0].length) {
            throw new RuntimeException("Length of X should be equal to length Y");
        }

        var pearson = 1.0; //calculatePearson(train_x, train_y);

        if (pearson < 0.8) {

            DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics(train_y);

            double median_predicted = descriptiveStatistics.getPercentile(50);

            return new Sextet<>(taskname, "LocallyJ", resourceToPredict, median_predicted, test_y[0], Math.abs((median_predicted - test_y[0]) / test_y[0]));
        }


        System.out.println(Arrays.deepToString(train_x).replaceAll("\\s+", ""));

        System.out.println(" " + Arrays.deepToString(train_x).replaceAll("\\s+", "") + " " + StringUtils.join(train_y, ',') + " " + Arrays.deepToString(test_x).replaceAll("\\s+", "") + " " + StringUtils.join(test_y, ','));

        ProcessBuilder processBuilder = new ProcessBuilder("python3", resolvePythonScriptPath("bayes_v2.py"), Arrays.deepToString(train_x).replaceAll("\\s+", ""), StringUtils.join(train_y, ','), Arrays.deepToString(test_x).replaceAll("\\s+", ""), StringUtils.join(test_y, ','));
        processBuilder.redirectErrorStream(true);

        double predicted = 0;

        try {
            Process process = processBuilder.start();
            process.waitFor();
            List<String> results = readProcessOutput(process.getInputStream());

            for (String s : results) {
                System.out.println(s);
                if (s.contains("Prediction:")) {
                    System.out.println(s);
                    predicted = Double.valueOf(s.split("\\[")[1].substring(0, s.split("\\[")[1].length() - 1)) * factor;
                }
            }

            int exitCode = 0;

            exitCode = process.waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new Sextet<>(taskname, "LocallyJ", resourceToPredict, predicted, test_y[0], Math.abs((predicted - test_y[0]) / test_y[0]));

    }

    public static double calculatePearson(double[] x, double[] y) {
        PearsonsCorrelation pearsonsCorrelation = new PearsonsCorrelation();

        return pearsonsCorrelation.correlation(x, y);
    }

    private static double calculateSpearman(double[] trainX, double[] trainY) {

        SpearmansCorrelation spearmansCorrelation = new SpearmansCorrelation();

        return spearmansCorrelation.correlation(trainX, trainY);

    }


    private static String resolvePythonScriptPath(String filename) {
        File file = new File("src/main/resources/" + filename);
        //File file = new File("" + filename);
        return file.getAbsolutePath();
    }

    private List<String> readProcessOutput(InputStream inputStream) throws IOException {
        try (BufferedReader output = new BufferedReader(new InputStreamReader(inputStream))) {
            return output.lines()
                    .collect(Collectors.toList());
        }
    }
}
