package be.yh

import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.util.*
import java.util.stream.Collectors
import kotlin.streams.asStream

class ZalandaMNISTLoader {
    private val MAGIC_OFFSET = 0
    private val OFFSET_SIZE = 4 //in bytes

    private val LABEL_MAGIC = 2049
    private val IMAGE_MAGIC = 2051

    private val NUMBER_ITEMS_OFFSET = 4
    private val ITEMS_SIZE = 4

    private val NUMBER_OF_ROWS_OFFSET = 8
    private val ROWS_SIZE = 4
    val ROWS = 28

    private val NUMBER_OF_COLUMNS_OFFSET = 12
    private val COLUMNS_SIZE = 4
    val COLUMNS = 28

    private val IMAGE_OFFSET = 16
    private val IMAGE_SIZE = ROWS * COLUMNS

    fun getDataSet(): MutableList<List<String>> {
        val labelsFile = File("C:\\Users\\yassin.hajaj\\IdeaProjects\\zalandomnistdl4jdemo\\src\\main\\resources\\fashion\\train-labels-idx1-ubyte")
        val imagesFile = File("C:\\Users\\yassin.hajaj\\IdeaProjects\\zalandomnistdl4jdemo\\src\\main\\resources\\fashion\\train-images-idx3-ubyte")

        val labelBytes = labelsFile.readBytes()
        val imageBytes = imagesFile.readBytes()

        val labelMagic = Arrays.copyOfRange(labelBytes, 0, OFFSET_SIZE)
        val imageMagic = Arrays.copyOfRange(imageBytes, 0, OFFSET_SIZE)

        if (ByteBuffer.wrap(labelMagic).int != LABEL_MAGIC) {
            throw IOException("Bad magic number in label file!")
        }

        if (ByteBuffer.wrap(imageMagic).int != IMAGE_MAGIC) {
            throw IOException("Bad magic number in image file!")
        }

        val numberOfLabels = ByteBuffer.wrap(Arrays.copyOfRange(labelBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).int
        val numberOfImages = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).int

        if (numberOfImages != numberOfLabels) {
            throw IOException("The number of labels and images do not match!")
        }

        val numRows = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_OF_ROWS_OFFSET, NUMBER_OF_ROWS_OFFSET + ROWS_SIZE)).int
        val numCols = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_OF_COLUMNS_OFFSET, NUMBER_OF_COLUMNS_OFFSET + COLUMNS_SIZE)).int

        if (numRows != ROWS && numCols != COLUMNS) {
            throw IOException("Bad image. Rows and columns do not equal " + ROWS + "x" + COLUMNS)
        }

        val list = mutableListOf<List<String>>()

        for (i in 0 until numberOfLabels) {
            val label = labelBytes[OFFSET_SIZE + ITEMS_SIZE + i]
            val imageData = Arrays.copyOfRange(imageBytes, i * IMAGE_SIZE + IMAGE_OFFSET, i * IMAGE_SIZE + IMAGE_OFFSET + IMAGE_SIZE)

            val imageDataList = imageData.iterator().asSequence().asStream().map { b -> b.toString() }.collect(Collectors.toList())
            imageDataList.add(label.toString())
            list.add(imageDataList)
        }
        return list
    }

}