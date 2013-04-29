package titanic.weka;
import java.io.File;
import java.io.IOException;
import java.util.Enumeration;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.core.converters.Loader;

/**
 * The Train class is responsible for loading the training data, instantiating a
 * Classifier, then building the classifier instance with the training data. It
 * then serializes the Classifier to disk for other operations to use.
 * 
 * As seen in the README.md file, we have converted the given CSV formatted
 * traiing and teat data into ARFF formatted files. This allows us to specify
 * the types of each column (nominal, numeric, string).
 * 
 * @author jbirchfield
 * 
 */
public class Train {

	public static void main(String[] args) throws Exception {
		/*
		 * First we load the training data from our ARFF file
		 */
		ArffLoader trainLoader = new ArffLoader();
		trainLoader.setSource(new File("train.arff"));
		trainLoader.setRetrieval(Loader.BATCH);
		Instances trainDataSet = trainLoader.getDataSet();

		/*
		 * Now we tell the data set which attribute we want to classify, in our
		 * case, we want to classify the first column: survived
		 */
		Attribute trainAttribute = trainDataSet.attribute(0);
		trainDataSet.setClass(trainAttribute);

		/*
		 * The RandomForest implementation cannot handle columns of type string,
		 * so we remove them for now.
		 */
		trainDataSet.deleteStringAttributes();

		/*
		 * Create a new Classifier of type RandomForest and configure it.
		 */
		RandomForest forest = new RandomForest();
		forest.setNumTrees(500);
		forest.setDebug(true);

		/*
		 * Now we train the classifier
		 */
		forest.buildClassifier(trainDataSet);

		/*
		 * We are done training the classifier, so now we serialize it to disk
		 */
		SerializationHelper.write("titanic.model", forest);
		System.out.println("Saved trained model to titanic.model");

	}
}
