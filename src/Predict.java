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

public class Predict {

	public static void main(String[] args) throws Exception {
		
		ArffLoader testLoader = new ArffLoader();
		testLoader.setSource(new File("test.arff"));
		testLoader.setRetrieval(Loader.BATCH);
		Instances testDataSet = testLoader.getDataSet();


		Attribute testAttribute = testDataSet.attribute(0);
		testDataSet.setClass(testAttribute);
		testDataSet.deleteStringAttributes();

		RandomForest forest = (RandomForest) SerializationHelper.read("titanic.model");

		ArffLoader test1Loader = new ArffLoader();
		test1Loader.setSource(new File("test.arff"));
		Instances test1DataSet = test1Loader.getDataSet();
		Attribute test1Attribute = test1DataSet.attribute(0);
		test1DataSet.setClass(test1Attribute);

		Enumeration testInstances = testDataSet.enumerateInstances();
		Enumeration test1Instances = test1DataSet.enumerateInstances();
		while (testInstances.hasMoreElements()) {
			Instance instance = (Instance) testInstances.nextElement();
			Instance instance1 = (Instance) test1Instances.nextElement();
			double classification = forest.classifyInstance(instance);
			instance1.setClassValue(classification);
		}

		CSVSaver predictedCsvSaver = new CSVSaver();
		predictedCsvSaver.setFile(new File("predict.csv"));
		predictedCsvSaver.setInstances(test1DataSet);
		predictedCsvSaver.writeBatch();
		
		System.out.println("Prediciton saved to predict.csv");
		
		

	}
}
