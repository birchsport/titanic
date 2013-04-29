import java.io.File;
import java.io.IOException;
import java.util.Enumeration;

import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.core.converters.Loader;

public class Verify {

	public static void main(String[] args) throws Exception {
		
		CSVLoader predictCsvLoader = new CSVLoader();
		predictCsvLoader.setSource(new File("predict.csv"));
		predictCsvLoader.setStringAttributes("3,8,10");
		predictCsvLoader.setNominalAttributes("1,4,11");
		Instances predictDataSet = predictCsvLoader.getDataSet();


		Attribute testAttribute = predictDataSet.attribute(0);
		predictDataSet.setClass(testAttribute);
		predictDataSet.deleteStringAttributes();

		RandomForest forest = (RandomForest) SerializationHelper.read("titanic.model");
		
		Evaluation evaluation = new Evaluation(predictDataSet);
		evaluation.evaluateModel(forest, predictDataSet, new Object[]{});
		
		System.out.println(evaluation.toSummaryString());


	}
}
