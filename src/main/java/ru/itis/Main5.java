/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package ru.itis;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**Example: Train a network to reproduce certain mathematical functions, and plot the results.
 * Plotting of the network output occurs every 'plotFrequency' epochs. Thus, the plot shows the accuracy of the network
 * predictions as training progresses.
 * A number of mathematical functions are implemented here.
 * Note the use of the identity function on the network output layer, for regression
 *
 * @author Alex Black
 */
public class Main5 {

    //Random number generator seed, for reproducability
    public static final int seed = 12345;
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 50;
    //How frequently should we plot the network output?
    private static final int plotFrequency = 15;
    //Number of data points
    private static final int nSamples = 1000;
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int batchSize = 100;
    //Network learning rate
    public static final double learningRate = 0.1;
    public static final Random rng = new Random(seed);
    public static final int numInputs = 1;
    private static final int numOutputs = 1;


    public static void main(final String[] args) throws IOException, InterruptedException {

        //Switch these two options to do different functions with different networks
        final MathFunction fn = new Pow2MathFunction();
        final MultiLayerConfiguration conf = getDeepDenseLayerNetworkConfiguration();

        //Generate the training data
        int numLinesToSkip = 1;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("mer_utf8_3.csv").getFile()));
        int batchSize = 63140;
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize, 1, 1, true);
        System.out.println(iterator.next());
        iterator.reset();

        final INDArray x = iterator.next().getFeatures();
        iterator.reset();
        final INDArray y = iterator.next().getLabels();
        //final DataSetIterator iterator = getTrainingData(x,fn,batchSize,rng);

        //Create the network
        final MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));


        //Train the network on the full data set, and evaluate in periodically
        final INDArray[] networkPredictions = new INDArray[nEpochs/ plotFrequency];
        for( int i=0; i<nEpochs; i++ ){
            System.out.println(i);
            iterator.reset();
            net.fit(iterator);
            if((i+1) % plotFrequency == 0) networkPredictions[i/ plotFrequency] = net.output(x, false);
        }
        iterator.reset();
        System.out.println(iterator.next().getFeatures());
        iterator.reset();
        System.out.println(net.output(iterator.next().getFeatures()));

        plot(fn, x, y, networkPredictions);
    }

    /** Returns the network configuration, 2 hidden DenseLayers of size 50.
     */
    private static MultiLayerConfiguration getDeepDenseLayerNetworkConfiguration() {
        final int numHiddenNodes = 5;
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.TANH).build())
                .layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.TANH).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();
    }

    /** Create a DataSetIterator for training
     * @param x X values
     * @param function Function to evaluate
     * @param batchSize Batch size (number of quickstartexamples for every call of DataSetIterator.next())
     * @param rng Random number generator (for repeatability)
     */
    @SuppressWarnings("SameParameterValue")
    private static DataSetIterator getTrainingData(final INDArray x, final MathFunction function, final int batchSize, final Random rng) {
        final INDArray y = function.getFunctionValues(x);
        final DataSet allData = new DataSet(x,y);

        final List<DataSet> list = allData.asList();
        Collections.shuffle(list,rng);
        DataSetIterator dataSetIterator = new ListDataSetIterator<>(list,batchSize);
        System.out.println(dataSetIterator.next());
        dataSetIterator.reset();
        return dataSetIterator;
    }

    private static void plot(final MathFunction function, final INDArray x, final INDArray y, final INDArray... predicted) {
        final XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet,x,y,"True Function (Labels)");

        System.out.println(predicted.length);
        for( int i=0; i<predicted.length; i++ ){
            addSeries(dataSet,x,predicted[i],String.valueOf(i));
        }

        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Regression Example - " + function.getName(),      // chart title
                "X",                        // x axis label
                function.getName() + "(X)", // y axis label
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

    private static void addSeries(final XYSeriesCollection dataSet, final INDArray x, final INDArray y, final String label){
        final double[] xd = x.data().asDouble();
        final double[] yd = y.data().asDouble();
        final XYSeries s = new XYSeries(label);
        for( int j=0; j<xd.length; j++ ) s.add(xd[j],yd[j]);
        dataSet.addSeries(s);
    }
}