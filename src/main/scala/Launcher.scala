import java.io.{BufferedWriter, File, FileWriter}
import java.nio.charset.Charset
import java.util.concurrent.atomic.AtomicInteger

import de.bwaldvogel.liblinear._
import org.apache.commons.csv.{CSVFormat, CSVParser, CSVRecord}
import org.slf4j.LoggerFactory

import scala.util.Try

//import org.deeplearning4j.nn.api.{ Layer, OptimizationAlgorithm }
//import org.deeplearning4j.nn.conf.distribution.UniformDistribution
//import org.deeplearning4j.nn.conf.layers.GravesLSTM
//import org.deeplearning4j.nn.conf.{ MultiLayerConfiguration, NeuralNetConfiguration, Updater }
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
//import org.deeplearning4j.nn.weights.WeightInit
//import org.deeplearning4j.optimize.listeners.ScoreIterationListener
//import org.nd4j.linalg.api.buffer.DoubleBuffer
//import org.nd4j.linalg.api.ndarray.INDArray
//import org.nd4j.linalg.dataset.DataSet
//import org.nd4j.linalg.factory.Nd4j
import spire.implicits.cfor

import scala.collection.mutable.{ArrayBuffer, ListBuffer, Map => MMap}

/**
  * @author yu
  */
object Launcher extends App {
  //
  //  def lstm(): Unit = {
  //
  //    def dsGen() = {
  //      implicit class DataSetAddFeatureVector(ds: DataSet) {
  //        def addFeatureVector(feature: Array[Double], example: Double) = {
  //          val dbf = new DoubleBuffer(feature.length)
  //          dbf.setData(feature)
  //          val ndarr: INDArray = Nd4j.zeros(feature.length)
  //          ndarr.setData(dbf)
  //          ds.addFeatureVector(ndarr, 1)
  //          ds
  //        }
  //      }
  //      val ds: DataSet = new DataSet()
  //      ds.addFeatureVector(Array[Double](0, 0), 0)
  //        .addFeatureVector(Array[Double](1, 1), 1)
  //        .addFeatureVector(Array[Double](-1, 1), 1)
  //    }
  //
  //    val lstmLayerSize = 20 //Number of units in each GravesLSTM layer
  //    val miniBatchSize = 3 //32 //Size of mini batch to use when  training
  //    val examplesPerEpoch = 5 * miniBatchSize //i.e., how many examples to learn on between generating samples
  //    val exampleLength = 10 //Length of each training example
  //    val numEpochs = 3 //Total number of training + sample generation epochs
  //    //val nSamplesToGenerate = 4 //Number of samples to generate after each training epoch
  //    //val nCharactersToSample = 300 //Length of each sample to generate
  //    //val generationInitialization: String = null //Optional character initialization; a random character is used if null
  //    // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
  //    // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
  //    val rng = new Random(12345)
  //
  //    //Get a DataSetIterator that handles vectorization of text into something we can use to train
  //    // our GravesLSTM network.
  //    val iter = dsGen().iterateWithMiniBatches() //getShakespeareIterator(miniBatchSize, exampleLength, examplesPerEpoch)
  //    val nOut: Int = iter.totalOutcomes()
  //
  //    //Set up network configuration:
  //    val conf: MultiLayerConfiguration =
  //      new NeuralNetConfiguration.Builder()
  //        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
  //        .learningRate(0.1)
  //        .rmsDecay(0.95)
  //        .seed(12345)
  //        .regularization(true)
  //        .l2(0.001)
  //        .list(3)
  //        .layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
  //          .updater(Updater.RMSPROP)
  //          .activation("tanh").weightInit(WeightInit.DISTRIBUTION)
  //          .dist(new UniformDistribution(-0.08, 0.08)).build())
  //        //        .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
  //        //          .updater(Updater.RMSPROP)
  //        //          .activation("tanh").weightInit(WeightInit.DISTRIBUTION)
  //        //          .dist(new UniformDistribution(-0.08, 0.08)).build())
  //        //        .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax") //MCXENT + softmax for classification
  //        //          .updater(Updater.RMSPROP)
  //        //          .nIn(lstmLayerSize).nOut(nOut).weightInit(WeightInit.DISTRIBUTION)
  //        //          .dist(new UniformDistribution(-0.08, 0.08)).build())
  //        .pretrain(false).backprop(true)
  //        .build()
  //
  //    val net = new MultiLayerNetwork(conf)
  //    net.init()
  //    net.setListeners(new ScoreIterationListener(1))
  //
  //    //Print the  number of parameters in the network (and for each layer)
  //    val layers: Array[Layer] = net.getLayers
  //    val totalNumParams = layers.zipWithIndex.map({
  //      case (layer, i) =>
  //        val nParams: Int = layer.numParams()
  //        println("Number of parameters in layer " + i + ": " + nParams)
  //        nParams
  //    }).sum
  //    println("Total number of network parameters: " + totalNumParams)
  //
  //    //Do training, and then generate and print samples from network
  //    (0 until numEpochs).foreach { i =>
  //      net.fit(iter)
  //
  //      println("--------------------")
  //      println("Completed epoch " + i)
  //      //      val samples: Array[String] = sampleCharactersFromNetwork(generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate)
  //      //      samples.indices.foreach { j =>
  //      //        println("----- Sample " + j + " -----")
  //      //        println(samples(j))
  //      //        println()
  //      //      }
  //      net.rnnClearPreviousState()
  //
  //      val feature: Array[Double] = Array(0, 0)
  //      val dbf = new DoubleBuffer(feature.length)
  //      dbf.setData(feature)
  //      val ndarr: INDArray = Nd4j.zeros(feature.length)
  //      ndarr.setData(dbf)
  //
  //      val output: INDArray = net.rnnTimeStep(ndarr)
  //      println("" + output.getDouble(0) + "/" + output.getDouble(1))
  //
  //      iter.reset() //Reset iterator for another epoch
  //    }
  //
  //    println("\n\nExample complete")
  //  }
  val logger = LoggerFactory.getLogger(Launcher.getClass)
  val pathData = "data"
  val pathTrain = s"$pathData/train.csv"
  val pathTest = s"$pathData/test.csv"
  val pathModel = s"$pathData/model"
  val pathResult = s"target/my_submit.csv"
  val model = trainModel()
  predict()

  def predict(): Unit = {
    val pw = new BufferedWriter(new FileWriter(new File(pathResult)))
    pw.append("PassengerId,Survived\n")
    ftLoad(pathTest, withTarget = false)._1.foreach { e =>
      val ft = e._1
      val id = e._2
      //val status = e._3.get
      val pred: Double = Linear.predict(model, ft)
      //println(s"cmp: $status vs. ${pred > 0} , ${if ((pred > 0) == status) "YES" else "NO"}")
      pw.append(id).append(",").append(if (pred > 0) "1" else "0").append("\n")
    }
    pw.flush()
    pw.close()
  }

  def trainModel() = {
    logger.info("start training...")

    val tdata: (Array[(Array[Feature], String, Option[Boolean])], Int) = ftLoad(pathTrain, withTarget = true)

    val px: Array[Array[Feature]] = tdata._1.map(_._1)
    val py = tdata._1.map(e => if (e._3.get) 1.0 else -1.0)

    val p = new Problem
    p.x = px
    p.y = py
    p.l = py.length
    p.n = tdata._2

    logger.info(s"loaded, examples: ${p.l}, features: ${p.n}")

    // -s 0
    // = 0 , 375 of 491 , cal: 0.7637474541751528
    val solver: SolverType = SolverType.L2R_LR
    // = 1,  381 of 491 , cal: 0.7759674134419552
    //val solver: SolverType = SolverType.L2R_L2LOSS_SVC_DUAL
    // = 2  381 of 491 , cal: 0.7759674134419552
    //val solver: SolverType = SolverType.L2R_L2LOSS_SVC
    // 377 of 491 , cal: 0.7678207739307535
    // 396 of 491 , cal: 0.8065173116089613
    //val solver: SolverType = SolverType.L2R_L1LOSS_SVC_DUAL
    // 377 of 491 , cal: 0.7678207739307535
    // 1, 0.0001 398 of 491 , cal: 0.8105906313645621
    //val solver: SolverType = SolverType.MCSVM_CS

    //val solver: SolverType = SolverType.L2R_L2LOSS_SVR_DUAL
    // cost of constraints violation
    val C: Double = 1
    // stopping criteria
    val eps: Double = 0.00001
    val parameter = new Parameter(solver, C, eps)

    logger.info("training model ... ")
    val model = Linear.train(p, parameter)
    logger.info("model trained ... ")
    model.save(new File(pathModel))

    // evaluate
    val acc = new AtomicInteger()
    val all = new AtomicInteger()
    tdata._1.foreach { e =>
      val ft = e._1
      val id = e._2
      val status = e._3.get
      val pred: Double = Linear.predict(model, ft)
      println(s"cmp: $status vs. ${pred > 0} , ${if ((pred > 0) == status) "YES" else "NO"}")
      all.getAndIncrement()
      if ((pred > 0) == status) acc.getAndIncrement()
    }
    println(s"status: ### ${acc.get()} of ${all.get()} , accuracy: ${acc.get().toDouble / all.get()}")


    model
  }

  def ftLoad(path: String, withTarget: Boolean = false) = {
    val csvit = CSVParser.parse(new File(path), Charset.defaultCharset(), CSVFormat.RFC4180).getRecords.iterator()
    //val abuff = ListBuffer[List[String]]()
    val abuff = ListBuffer[(Array[Feature], String, Option[Boolean])]()
    csvit.next()
    var ylen = 0
    while (csvit.hasNext) {
      val elem: CSVRecord = csvit.next()
      val lbuff = ListBuffer[String]()
      cfor(0)(_ < elem.size(), _ + 1) { i =>
        lbuff.append(elem.get(i))
      }
      if (lbuff.nonEmpty) {
        val fta = ArrayBuffer[Feature]()
        val id = lbuff.head
        val survived = if (withTarget) Option(lbuff(1) == "1") else None
        val offset = new AtomicInteger(if (withTarget) 2 else 1)
        var cIndex = 1
        //Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        val pclass = lbuff(offset.getAndIncrement()).toInt // 2 / 1
        fta.append(new FeatureNode(cIndex + pclass - 1, 1.0))
        cIndex += 3
        val name = lbuff(offset.getAndIncrement()).toLowerCase // 3 / 2
        fta.append(new FeatureNode(cIndex, if (name.contains("dr.")) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (name.contains("mr.")) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (name.contains("mrs.")) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (name.contains("miss.")) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (name.contains("master.")) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex,
          if (name.contains("master.") || name.contains("dr.")) 1.0 else 0.0))
        cIndex += 1

        val sex = lbuff(offset.getAndIncrement()).toLowerCase // 3 / 2
        fta.append(new FeatureNode(cIndex, if (sex == "male") 1.0 else -1.0))
        cIndex += 1
        //fta.append(new FeatureNode(cIndex, if (sex == "female") 1.0 else 0.0))
        //cIndex += 1

        val age = Try(lbuff(offset.getAndIncrement()).toInt).getOrElse(0)
        fta.append(new FeatureNode(cIndex, age.toDouble))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (age > 0 && age < 3) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (age >= 3 && age < 10) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (age >= 10 && age < 15) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (age >= 15 && age < 20) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (age >= 20 && age < 40) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (age >= 40 && age < 60) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (age >= 60) 1.0 else 0.0))
        cIndex += 1

        val sibSp = lbuff(offset.getAndIncrement()).toInt
        fta.append(new FeatureNode(cIndex, if (sibSp == 0) 1.0 else -1.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (sibSp == 1) 1.0 else -1.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (sibSp == 2) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (sibSp == 3) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (sibSp > 3) 1.0 else -1.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (sibSp > 2) 1.0 else -1.0))
        cIndex += 1

        val parch = lbuff(offset.getAndIncrement()).toInt
        fta.append(new FeatureNode(cIndex, if (parch == 0) 1.0 else -1.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (parch == 1) 1.0 else -1.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (parch == 2) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (parch == 3) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (parch > 3) 1.0 else -1.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (parch > 2) 1.0 else -1.0))
        cIndex += 1

        val families = sibSp + parch
        fta.append(new FeatureNode(cIndex, if (families == 0) 1.0 else -1.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (families == 1) 1.0 else -1.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (families == 2) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (families == 3) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (families > 3) 1.0 else -1.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (families > 2) 1.0 else -1.0))
        cIndex += 1


        //TODO tickets?
        val ticket = Try(lbuff(offset.getAndIncrement()).filter(c => c >= '0' && c <= '9').toInt)

        ticket.getOrElse(-1) match {
          case -1 =>
          case v: Int =>
            //fta.append(new FeatureNode(cIndex, v.toDouble))
        }
        cIndex += 1

        val fare = Try(lbuff(offset.getAndIncrement()).toDouble).getOrElse(0.0)
        //fta.append(new FeatureNode(cIndex, fare))
        //cIndex += 1
        fta.append(new FeatureNode(cIndex, if (fare < 10) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (fare >= 10 && fare < 50) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (fare >= 50) 1.0 else 0.0))
        cIndex += 1

        val cabin = lbuff(offset.getAndIncrement()).toLowerCase
        fta.append(new FeatureNode(cIndex, if (cabin.contains("a")) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (cabin.contains("b")) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (cabin.contains("c")) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (cabin.contains("d")) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (cabin.contains("e")) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (cabin.contains("f")) 1.0 else 0.0))
        cIndex += 1
        fta.append(new FeatureNode(cIndex, if (cabin.contains("g")) 1.0 else 0.0))
        cIndex += 1

        val embarked = lbuff(offset.getAndIncrement()).toLowerCase
        if (embarked.length > 0) {
          val c = embarked.head
          fta.append(new FeatureNode(cIndex, if (c == 'c') 1.0 else 0.0))
          cIndex += 1
          fta.append(new FeatureNode(cIndex, if (c == 'q') 1.0 else 0.0))
          cIndex += 1
          fta.append(new FeatureNode(cIndex, if (c == 's') 1.0 else 0.0))
          cIndex += 1
        }
        cIndex += 3
        ylen = cIndex
        abuff.append((fta.toArray, id, survived))
      }
      //abuff.append(lbuff.toList)
    }
    (abuff.toArray, ylen)
  }
}
