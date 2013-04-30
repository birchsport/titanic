package titanic.weka.multi;

import java.io.File;

import weka.classifiers.trees.BFTree;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.NBTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
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
		MultiClassifier multiClassifier = new MultiClassifier();
		
		 RandomForest forest = new RandomForest();
		 forest.setNumTrees(500);
		 forest.setDebug(true);
		 
		 BFTree bfTree = new BFTree();
		 bfTree.setNumFoldsPruning(10);
		 bfTree.setHeuristic(true);
		 bfTree.setUseGini(true);
		 bfTree.setDebug(true);
		 
		 NBTree nbTree = new NBTree();
		 nbTree.setDebug(true);
		 
		 J48 j48 = new J48();
		 j48.setDebug(true);
		 j48.setNumFolds(10);
		 
		 LMT lmt = new LMT();
		 lmt.setDebug(true);
		 
		 multiClassifier.addClassifier(forest);
		 multiClassifier.addClassifier(bfTree);
		 multiClassifier.addClassifier(nbTree);
		 multiClassifier.addClassifier(j48);
		 multiClassifier.addClassifier(lmt);

		/*
		 * Now we train the classifier
		 */
		multiClassifier.buildClassifier(trainDataSet);

		/*
		 * We are done training the classifier, so now we serialize it to disk
		 */
		SerializationHelper.write("titanic-multi.model", multiClassifier);
		System.out.println("Saved trained model to titanic-multi.model");

	}
}
