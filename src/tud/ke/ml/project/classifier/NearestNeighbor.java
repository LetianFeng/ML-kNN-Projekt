package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import tud.ke.ml.project.util.Pair;

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
		throw new NotImplementedException();
	}

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		return this.getWinner(this.getUnweightedVotes(subset));
	}

	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> data) {
		throw new NotImplementedException();
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
