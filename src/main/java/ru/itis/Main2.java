package ru.itis;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.swing.*;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class Main2 {
    public static void main(String[] args) throws IOException, InterruptedException {
        int numLinesToSkip = 1;
        char delimiter = ';';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("data.csv").getFile()));

        int labelIndex = 3;
        int numClasses = 19;
        int batchSize = 63141;

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fitLabel(true);
        normalizer.fit(trainingData);
        normalizer.fit(testData);
        normalizer.transform(testData);
        normalizer.transform(trainingData);

        final int numInputs = 3;
        int outputNum = 19;
        long seed = 12345;

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .miniBatch(false)
                .seed(seed)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.01, 0.9))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(50)
                        .build())
                .layer( new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(50).nOut(outputNum).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(1));

        for(int i = 0; i < 150; i++ ) {
            System.out.println(i);
            model.fit(trainingData);
        }

        Evaluation eval = new Evaluation(19);
        INDArray output = model.output(testData.getFeatures(), false);
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());

        plot(toArgMaxINDArray(testData.getLabels()).toDoubleVector(),
                toArgMaxINDArray(testData.getLabels()).toDoubleVector(),
                toArgMaxINDArray(output).toDoubleVector());

        userInput(model);
    }

    private static void plot(double[] x, double[] y, double[] predicted) {
        final XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet,x,y,"True Function (Labels)");



        addSeries(dataSet,x,predicted,"predicted");


        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Classes Example - " ,      // chart title
                "X",                        // x axis label
                "(X)", // y axis label
                dataSet,                    // data
                PlotOrientation.VERTICAL,
                true,                       // include legend
                true,                       // tooltips
                false                       // urls
        );

        final ChartPanel panel = new ChartPanel(chart);

        final JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
    }

    private static void addSeries(final XYSeriesCollection dataSet, double[] x, double[] y, final String label){
        final double[] xd = x;
        final double[] yd = y;
        final XYSeries s = new XYSeries(label);

        for( int j=0; j<xd.length; j++ )
            s.add(xd[j],yd[j]);
        dataSet.addSeries(s);
    }

    private static INDArray toArgMaxINDArray(INDArray indArray) {
        INDArray result = Nd4j.create(indArray.rows());

        for(int i = 0; i < indArray.rows(); i++) {
            INDArray row = indArray.getRow(i);
            result.put(i, Nd4j.create(row.argMax().toDoubleVector()));
        }

        return  result;
    }

    public static void userInput(MultiLayerNetwork model) {
        double[] a = new double[3];
        System.out.println("Введите время работы, время простоя, коэфф. производительности:");
        Scanner scanner = new Scanner(System.in);
        INDArray input;
        while(true) {
            a[0] = scanner.nextDouble();
            a[1] = scanner.nextDouble();
            a[2] = scanner.nextDouble();

            input = Nd4j.create(a, 1, 3);
            System.out.println(input);
            System.out.println(toArgMaxINDArray(model.output(input)));
        }
    }
}
