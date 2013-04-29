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

public class Train {

	public static void main(String[] args) throws Exception {
		ArffLoader trainLoader = new ArffLoader();
		trainLoader.setSource(new File("train.arff"));
		trainLoader.setRetrieval(Loader.BATCH);
		Instances trainDataSet = trainLoader.getDataSet();

		Attribute trainAttribute = trainDataSet.attribute(0);
		trainDataSet.setClass(trainAttribute);
		trainDataSet.deleteStringAttributes();

		RandomForest forest = new RandomForest();
		forest.setNumTrees(500);
		forest.setDebug(true);
		forest.buildClassifier(trainDataSet);

		SerializationHelper.write("titanic.model", forest);
		System.out.println("Saved trained model to titanic.model");

	}
}
