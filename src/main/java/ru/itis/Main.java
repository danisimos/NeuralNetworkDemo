package ru.itis;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException, InterruptedException {
        int numLinesToSkip = 0;
        char delimiter = ';';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("mer_utf8_3.csv").getFile()));

        //int labelIndex = 4;
        //int numClasses = 6;
        int batchSize = 63140;

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize, 1, 1, true);
        DataSet allData = iterator.next();
        System.out.println(allData.getFeatures());
        System.out.println("-----------------------------------------------------------------------------------------------------");
        //allData.shuffle();
        //SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.85);

        //DataSet trainingData = testAndTrain.getTrain();
        //DataSet testData = testAndTrain.getTest();
        //DataNormalization normalizer = new NormalizerStandardize();
        //normalizer.fitLabel(true);
        //normalizer.fit(iterator);
        //iterator.setPreProcessor(normalizer);
        //normalizer.fit(trainingData);
        //normalizer.fit(testData);
        //normalizer.transform(trainingData);
        //normalizer.transform(testData);
        //normalizer.transform(allData);
        //normalizer.transformLabel(allData.getLabels());

        final int numInputs = 3;
        int outputNum = 1;
        long seed = 6;

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .miniBatch(false)

                //.activation(Activation.TANH)
                //.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                //.updater(new Sgd(0.1))
                .updater(new Nesterovs(0.00001, 0.9))
                //.l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(1).nOut(1)
                        .activation(Activation.IDENTITY)
                        .build())
                /*.layer(new DenseLayer.Builder().nIn(3).nOut(3)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new DenseLayer.Builder().nIn(3).nOut(3)
                        .activation(Activation.IDENTITY)
                        .build())*/
                .layer( new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(1).nOut(1).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(1));

        for(int i=0; i<1000; i++ ) {
            model.fit(iterator);
            iterator.reset();
        }

        INDArray predictionOutput = model.output(Nd4j.create(new double[]{24}, 1, 1));
        INDArray predictionOutput2 = model.output(Nd4j.create(new double[]{691}, 1, 1));
        INDArray predictionOutput3 = model.output(Nd4j.create(new double[]{528}, 1, 1));
        INDArray predictionOutput4 = model.output(Nd4j.create(new double[]{732}, 1, 1));
        INDArray predictionOutput5 = model.output(Nd4j.create(new double[]{0}, 1, 1));
        INDArray predictionOutput6 = model.output(allData.getFeatures());
        System.out.println(predictionOutput);
        System.out.println(predictionOutput2);
        System.out.println(predictionOutput3);
        System.out.println(predictionOutput4);
        System.out.println(predictionOutput5);
        System.out.println(predictionOutput6);
        //System.out.println(Nd4j.create(new double[]{24.0000, 0, 696.0000, 0, 4.0000}, 1,5));
        //System.out.println(Nd4j.create(new double[]{24.0000, 0, 696.0000, 0}, 3,1));
        //Evaluation eval = new Evaluation();
        //eval.eval(allData.getLabels(), predictionOutput6);
        //System.out.println(eval.stats());

        //System.out.println(predictionOutput);

        /*Evaluation eval = new Evaluation(6);
        //System.out.println(iterator.);
        INDArray output = model.output(testData.getFeatures());
        //System.out.println("asdasdasdasdasdasdasdasdasdasd");
        //System.out.println(output.getRow(0));
        //System.out.println(testData.getFeatures());

        eval.eval(testData.getLabels(), output);

        System.out.println(eval.stats());*/
    }
}
