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
		throw new NotImplementedException();
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
		throw new NotImplementedException();
	}

	@Override
	protected Object getWinner(Map<Object, Double> votes) {
	    double maxVote = 0;
	    Object maxObject = null;
	    for (Map.Entry<Object, Double> entry : votes.entrySet()) {
	        if (maxObject == null) {
	            maxVote = entry.getValue();
	            maxObject = entry.getKey();
            }

            if (entry.getValue() > maxVote) {
                maxVote = entry.getValue();
                maxObject = entry.getKey();
            }
        }

        return maxObject;
	}

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		return this.getWinner(this.getUnweightedVotes(subset));
	}

	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> data) {
		List<Pair<List<Object>, Double>> distances = new ArrayList<>();
		// determineManhattanDistance
        for (List<Object> instance2 : this.traindata)
            if (!data.equals(instance2))
                distances.add(new Pair<List<Object>, Double>(instance2, this.determineManhattanDistance(data, instance2)));

		// sort list distances
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
		throw new NotImplementedException();
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
