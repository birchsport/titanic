package titanic.weka.multi;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class MultiClassifier extends Classifier {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8996052328536347834L;
	
	List<Classifier> classifiers = new ArrayList<Classifier>();

	@Override
	public void buildClassifier(Instances data) throws Exception {
		for (Classifier classifier : classifiers) {
			classifier.buildClassifier(data);
		}
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		double sum = 0.0;
		for (Classifier classifier : classifiers) {
			double classification = classifier.classifyInstance(instance);
			sum += classification;
		}
		if(sum >= classifiers.size()/2) {
			return 1.0;
		}
		return 0.0;
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		return super.clone();
	}

	public void addClassifier(Classifier classifier) {
		classifiers.add(classifier);
	}
	
	@Override
	public String toString() {
		StringBuffer buffer = new StringBuffer();
		buffer.append("MultiClassifier\n");
		buffer.append("-----------------\n");
		for (Classifier classifier : classifiers) {
			buffer.append(classifier.toString());
			buffer.append("\n-----------------\n");
		}
		
		return buffer.toString();
	}

}
