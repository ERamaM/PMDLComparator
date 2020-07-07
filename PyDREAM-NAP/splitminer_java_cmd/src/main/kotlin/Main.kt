import au.edu.qut.processmining.miners.splitminer.SplitMiner
import au.edu.qut.processmining.miners.splitminer.ui.dfgp.DFGPUIResult
import au.edu.qut.processmining.miners.splitminer.ui.miner.SplitMinerUIResult
import com.raffaeleconforti.context.FakePluginContext
import com.raffaeleconforti.conversion.bpmn.BPMNToPetriNetConverter
import com.raffaeleconforti.conversion.petrinet.PetriNetToBPMNConverter
import com.raffaeleconforti.marking.MarkingDiscoverer
import com.raffaeleconforti.wrappers.PetrinetWithMarking
import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.DefaultHelpFormatter
import com.xenomachina.argparser.default
import com.xenomachina.argparser.mainBody
import edu.uic.prominent.processmining.decaypns.log.util.XLogReader
import edu.uic.prominent.processmining.decaypns.prom.model.ModelUtils
import org.deckfour.xes.classification.XEventNameClassifier
import org.deckfour.xes.model.XLog
import org.processmining.models.graphbased.directed.bpmn.BPMNDiagram
import java.io.FileDescriptor
import java.io.FileOutputStream
import java.io.PrintStream
import java.nio.file.Paths
import java.time.Instant
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class Arguments(parser : ArgParser) {
    val log_name by parser.storing("-l", "--log", help="base name process log in xes.gz format")
    val model_folder by parser.storing("-m", "--model", help="folder where models are to be saved")
    val n_threads by parser.storing("-t", "--threads", help="number of threads to use") { toInt() }.default(8)
    val best_model_folder by parser.storing("-b", "--best", help="best model folder")
}

fun main(args: Array<String>) {
    mainBody{
        ArgParser(args, helpFormatter = DefaultHelpFormatter()).parseInto(::Arguments).run {
            var train_log_name: String
            var folder_model_name: String

            train_log_name = this.log_name
            folder_model_name = this.model_folder

            val eps = ArrayList<Double>()
            val eta = ArrayList<Double>()

            for (i in 0..10) {
                eps.add(i / 10.0)
                eta.add(i / 10.0)
            }

            // Load log using citius code
            val n_threads = this.n_threads

            var executor = Executors.newFixedThreadPool(n_threads)
            for (e in eps) {
                for (et in eta) {
                    val thread = Runnable {
                        println("Model: $e $et")
                        mine_and_save(train_log_name, folder_model_name, e, et)
                    }
                    executor.execute(thread)
                }
            }
            executor.shutdown()
            executor.awaitTermination(1, TimeUnit.DAYS)

            /*
            var bm = best_model_folder
            if (!bm.contains("/")){
                bm += "/"
            }
            val executionLog = FileWriter(
                bm +
                        this.log_name.replace("/", "_").replace(".xes.gz", "_") + "model_logs.txt"
            )
            // Export pnml
            var resultMap = HashMap<String, Double>()
            for (e in eps) {
                for (et in eta) {
                    print("Processing: $e $et\n")
                    val model_name = get_filename(log_name)
                    val mined_output = Paths.get(model_folder, model_name + "_" + "$e" + "_" + "$et").toString()
                    val log = load_log(train_log_name)
                    val model = PNMLFileParser("$mined_output.pnml").read()

                    val fitness = AlignmentBasedFitness().compute(model, log)
                    // val precision = AdvancedBehaviouralAppropriateness().compute(model, log)
                    // print("Precision: $precision")
                    println("Model $mined_output with fitness $fitness and precision (coverage): TBD")
                    executionLog.append("Model $mined_output with fitness $fitness\n")
                    resultMap[mined_output] = fitness
                    executionLog.flush()
                }
            }

            val sortedMap = resultMap.toList().sortedBy { (k, v) -> v }.toMap()
            val bestmodel = sortedMap.keys.last()
            // Copy model to best_models directory
            var filename = get_filename(bestmodel)
            File("$bestmodel.pnml").copyTo(File("$bm$filename.pnml"))
            */
    }
}
}

private fun mine_and_save(log_name: String, model_folder: String, e: Double, et: Double) : String {
    // We need to do the WHOLE process for each thread to avoid concurrent modification exception
    val log = load_log(log_name)
    val miner = SplitMiner()

    val classifier = XEventNameClassifier()
    val context = FakePluginContext()

    val model_name = get_filename(log_name)

    val mined_output = Paths.get(model_folder, model_name + "_" + "$e" + "_" + "$et").toString()
    // do some dark magic to access rafaelle conforti code and use prom code without ui
    // use the same parameters as DREAM-NAP
    val bpmn = miner.mineBPMNModel(
        log,
        classifier,
        e,
        et,
        DFGPUIResult.FilterType.WTH,
        true,
        true,
        false,
        SplitMinerUIResult.StructuringTime.NONE
    )
    // bpmn to petrinet
    val petrinet = convertToPetrinet(context, bpmn)
    // petrinet to pnml
    ModelUtils.exportPetrinet(context, petrinet, "$mined_output.pnml")
    return "$mined_output.pnml"
}

private fun get_filename(bestmodel: String): String {
    var filename = bestmodel
    // If its a route, get the filename
    if (bestmodel.contains("/")) {
        val splits = bestmodel.split("/")
        filename = splits[splits.size - 1]
    }
    return filename
}

fun load_log(log_name : String) : XLog {
    return XLogReader.openLog(log_name)
}

fun convertToPetrinet(
    context: org.processmining.contexts.uitopia.UIPluginContext,
    diagram: BPMNDiagram
): PetrinetWithMarking {
    val result: Array<Any?> = BPMNToPetriNetConverter.convert(diagram)
    if (result[1] == null) result[1] =
        PetriNetToBPMNConverter.guessInitialMarking(result[0] as org.processmining.models.graphbased.directed.petrinet.Petrinet?)
    if (result[2] == null) result[2] =
        PetriNetToBPMNConverter.guessFinalMarking(result[0] as org.processmining.models.graphbased.directed.petrinet.Petrinet?)
    if (result[1] == null) result[1] = MarkingDiscoverer.constructInitialMarking(
        context,
        result[0] as org.processmining.models.graphbased.directed.petrinet.Petrinet?
    ) else MarkingDiscoverer.createInitialMarkingConnection(
        context,
        result[0] as org.processmining.models.graphbased.directed.petrinet.Petrinet?,
        result[1] as org.processmining.models.semantics.petrinet.Marking?
    )
    if (result[2] == null) result[2] = MarkingDiscoverer.constructFinalMarking(
        context,
        result[0] as org.processmining.models.graphbased.directed.petrinet.Petrinet?
    ) else MarkingDiscoverer.createFinalMarkingConnection(
        context,
        result[0] as org.processmining.models.graphbased.directed.petrinet.Petrinet?,
        result[1] as org.processmining.models.semantics.petrinet.Marking?
    )
    System.setOut(PrintStream(FileOutputStream(FileDescriptor.out)))
    return PetrinetWithMarking(
        result[0] as org.processmining.models.graphbased.directed.petrinet.Petrinet?,
        result[1] as org.processmining.models.semantics.petrinet.Marking?,
        result[2] as org.processmining.models.semantics.petrinet.Marking?
    )
}


