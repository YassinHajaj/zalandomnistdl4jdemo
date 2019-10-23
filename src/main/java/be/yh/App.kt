package be.yh

import org.datavec.api.records.reader.impl.collection.ListStringRecordReader
import org.datavec.api.split.ListStringSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import java.io.File
import java.util.concurrent.TimeUnit

object App {

    @JvmStatic
    fun main(args: Array<String>) {
        val uiServer = UIServer.getInstance()
        val inMemoryStatsStorage = InMemoryStatsStorage()
        uiServer.attach(inMemoryStatsStorage)

        val dataset = ZalandaMNISTLoader().getDataSet()
        val trainDatasetIterator = createDatasetIterator(dataset.subList(0, 500))
        val testDatasetIterator = createDatasetIterator(dataset.subList(500, 600))

        val cnnConfig = buildCNN()

        val cnn = MultiLayerNetwork(cnnConfig)
        cnn.init()
        cnn.setListeners(StatsListener(inMemoryStatsStorage))
        regularLearning(cnn, trainDatasetIterator, testDatasetIterator)
    }

    private fun createDatasetIterator(dataset: MutableList<List<String>>): RecordReaderDataSetIterator {
        val listStringRecordReader = ListStringRecordReader()
        listStringRecordReader.initialize(ListStringSplit(dataset))
        return RecordReaderDataSetIterator(listStringRecordReader, 128, 28 * 28, 10)
    }

    private fun regularLearning(cnn: MultiLayerNetwork, trainDatasetIterator: RecordReaderDataSetIterator, testDatasetIterator: RecordReaderDataSetIterator) {
        for (i in 0 until 10) {
            println("Epoch $i")
            cnn.fit(trainDatasetIterator)
        }

        val evaluation = Evaluation(10)
        while (testDatasetIterator.hasNext()) {
            val next = testDatasetIterator.next()
            val output = cnn.output(next.features)
            evaluation.eval(next.labels, output)
        }

        println(evaluation.stats())
        println(evaluation.confusionToString())
    }

    private fun earlyStoppingLearning(testDatasetIterator: RecordReaderDataSetIterator, cnnConfig: MultiLayerConfiguration, trainDatasetIterator: RecordReaderDataSetIterator) {
        val saveDirectory = System.getProperty("current.dir") + "/DL4JEarlyStoppingModels"
        File(saveDirectory).mkdir()

        val saver = LocalFileModelSaver(saveDirectory)

        val esConf = EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(MaxEpochsTerminationCondition(5))
                .iterationTerminationConditions(MaxTimeIterationTerminationCondition(5, TimeUnit.MINUTES))
                .scoreCalculator(DataSetLossCalculator(testDatasetIterator, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(saver)
                .build()

        val earlyStoppingTrainer = EarlyStoppingTrainer(esConf, cnnConfig, trainDatasetIterator)
        val result = earlyStoppingTrainer.fit()

        println("Termination reason: " + result.terminationReason)
        println("Termination details: " + result.terminationDetails)
        println("Total epochs: " + result.totalEpochs)
        println("Best epoch number: " + result.bestModelEpoch)
        println("Score at best epoch: " + result.bestModelScore)
        println(result.scoreVsEpoch)
    }

    private fun buildCNN(): MultiLayerConfiguration {
        // conv1
        // conv2
        // pooling
        // dense
        // batchn
        // outputlayer
        // neuralnet
    }

    private fun buildMultiLayerModel(): MultiLayerConfiguration? {
        return NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(Adam())
                .list()
                .layer(0, DenseLayer.Builder()
                        .nIn(28 * 28)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build()
                )
                .layer(1, OutputLayer.Builder()
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build()
                )
                .build()
    }
}
