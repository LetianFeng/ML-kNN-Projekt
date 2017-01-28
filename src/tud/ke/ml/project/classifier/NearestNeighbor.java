package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.*;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import tud.ke.ml.project.util.Pair;
import weka.core.ListOptions;

/**
 * This implementation assumes the class attribute is always available (but probably not set).
 * 
 */
public class NearestNeighbor extends INearestNeighbor implements Serializable {
	private static final long serialVersionUID = 1L;

	protected double[] scaling;
	protected double[] translation;

	private List<List<Object>> traindata;

	@Override
	public String getMatrikelNumbers() {
		return "2441890,2255840,xxxxxx";
	}

	@Override
	protected void learnModel(List<List<Object>> data) {
		this.traindata = data;
	}

	@Override
	protected Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset) {

	    Map<Object, Double> result = new HashMap<>();

	    // get class attribute in nearest train data & put into the result map
	    for (Pair<List<Object>, Double> pair : subset){
	        Object classAttribute = pair.getA().get(this.getClassAttribute());

	        // if class attribute doesn't exist in  the result map
	        if (!result.containsKey(classAttribute))
	            result.put(classAttribute, 0.0);
	        // if already existed in the map, then add 1 to the entry-value
	        else
	            result.put(classAttribute, result.get(classAttribute) + 1);
        }

        return result;
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
		throw new NotImplementedException();
	}

	@Override
	protected Object getWinner(Map<Object, Double> votes) {
	    double maxVote = 0;
	    Object maxAttribute = null;

	    // find the class attribute with biggest vote in the map
	    for (Map.Entry<Object, Double> entry : votes.entrySet()) {
	        // save first attribute & vote in maxAttribute & maxVote
            // or if a entry has larger vote, update them
            if (maxAttribute == null || entry.getValue() > maxVote) {
                maxVote = entry.getValue();
                maxAttribute = entry.getKey();
            }
        }

        return maxAttribute;
	}

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
	    // select method to get votes depending on if it is inverse weighting
		if (this.isInverseWeighting())
			return this.getWinner(this.getWeightedVotes(subset));
		else
			return this.getWinner(this.getUnweightedVotes(subset));
	}

	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> data) {

		List<Pair<List<Object>, Double>> distances = new ArrayList<>();

		// determineManhattanDistance
        for (List<Object> trainInstance : this.traindata)
            if (!data.equals(trainInstance))
                distances.add(new Pair<>(trainInstance, this.determineManhattanDistance(trainInstance, data)));

		// sort the list distances
		Collections.sort(distances, new Comparator<Pair<List<Object>, Double>>() {
            @Override
            public int compare(Pair<List<Object>, Double> pair1, Pair<List<Object>, Double> pair2) {
                if (pair1.getB() == pair2.getB())
                    return 0;
                if (pair1.getB() > pair2.getB())
                    return 1;
                return -1;
            }
        });

		// remove the last element till the list has the size as getkNearest()
		while (distances.size() > this.getkNearest())
			distances.remove(distances.size() - 1);

		return  distances;
	}

	@Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {

	    double distance = 0.0;
        List<Object> trainInstance = new ArrayList<>();
        List<Object> testInstance = new ArrayList<>();

        // remove class attribute from train instance
        if (instance1.size() > instance2.size()) {
            trainInstance.addAll(instance1);
            trainInstance.remove(this.getClassAttribute());

            testInstance.addAll(instance2);
        } else if (instance1.size() < instance2.size()) {
            trainInstance.addAll(instance2);
            trainInstance.remove(this.getClassAttribute());

            testInstance.addAll(instance1);
        } else {
            trainInstance.addAll(instance1);
            testInstance.addAll(instance2);
        }

        for (int i = 0; i < trainInstance.size(); i++) {
            Object trainAttribute = trainInstance.get(i);
            Object testAttribute = testInstance.get(i);

            // if symbolic attribute, use 0/1 distance, otherwise abs(v1 -v2)
            if (trainAttribute instanceof String) {
                if (!trainAttribute.equals(testAttribute))
                    distance += 1;
            } else {
                distance += Math.abs((Double)trainAttribute - (Double)testAttribute);
            }
        }

		return distance;
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
		throw new NotImplementedException();
	}

	@Override
	protected double[][] normalizationScaling() {
		throw new NotImplementedException();
	}

}
