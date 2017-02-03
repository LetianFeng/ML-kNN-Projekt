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
		return "2441890,2255840,2571142";
	}

	@Override
	protected void learnModel(List<List<Object>> data) {
	    // save the data in the object as an attribute
		this.traindata = data;
	}

	@Override
	protected Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset) {
        // initialize a map for result
        // its entry is <class attribute, vote>
	    Map<Object, Double> result = new HashMap<>();

	    // get all kind of class attributes in the nearest train data subset
        // put them into the result map with their votes
	    for (Pair<List<Object>, Double> pair : subset){
	        // get class attribute of an nearest instance
	        Object classAttribute = pair.getA().get(this.getClassAttribute());

	        // if class attribute doesn't exist in  the result map
            // put it into the map with the value vote 1
	        if (!result.containsKey(classAttribute))
	            result.put(classAttribute, 1.0);
	        // if already existed in the map, then add 1 to the existing vote
	        else
	            result.put(classAttribute, result.get(classAttribute) + 1);
        }

        // return the map
        return result;
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
        // initialize a map for result
        // its entry is <class attribute, vote>
		Map<Object, Double> result = new HashMap<>();

		// initialize 2 local variable for calculating the weight to be added on the vote
		double distance;
		double weight;


        // get all kind of class attributes in the nearest train data subset
        // put them into the result map with their votes
		for (Pair<List<Object>, Double> pair : subset){
            // get class attribute of an nearest instance
			Object classAttribute = pair.getA().get(this.getClassAttribute());

			// get distance of an instance
			distance = pair.getB();

			// because of double calculation in java can be inprecise
            // so distances less than 0.0000001 are considered as 0
			if (distance < 0.0000001)
			    // for distance 0, add fixed weight 999.99
				weight = 999.99;
			else
			    // for other distances, add weight of inverse distance
				weight = 1 / distance;

			// if class attribute doesn't exist in the result map
            // put it into the map with the value weight
			if (!result.containsKey(classAttribute))
				result.put(classAttribute, weight);

			// if already existed in the map, then add weight to the existing vote
			else
				result.put(classAttribute, result.get(classAttribute) + weight);
		}

		// return the map
		return result;
	}

	@Override
	protected Object getWinner(Map<Object, Double> votes) {
	    // initialize local variable for max vote & corresponding attribute
	    double maxVote = 0;
        Object maxAttribute = null;
        // for the case of multiple attributes with the same max vote
        // initialize a list, in which all attributes with the max vote are saved
	    List<Object> winners = new LinkedList<>();

	    // find the class attribute(s) with max vote in the map
	    for (Map.Entry<Object, Double> entry : votes.entrySet()) {
	        // save first attribute & vote in maxAttribute & maxVote
            if (maxAttribute == null) {
                maxVote = entry.getValue();
                maxAttribute = entry.getKey();
                winners.add(entry.getKey());
            // for following attributes
            } else {
                // attribute with less vote than max vote, do nothing
                if (entry.getValue() < maxVote)
                    continue;
                // attribute with larger vote than max vote
                // or equal vote as the max vote
                else {
                    // larger vote, clean the list, update max vote
                    // after that, add attribute to list
                    if (entry.getValue() > maxVote) {
                        maxVote = entry.getValue();
                        maxAttribute = entry.getKey();
                        winners = new LinkedList<>();
                    }
                    // equal vote, only add attribute to list
                    winners.add(entry.getKey());
                }
            }
        }

        // if multiple max attributes, randomly select one & return
        // otherwise return directly
        if (winners.size() > 1) {
	        Random rand = new Random();
	        int n = rand.nextInt(winners.size());
	        maxAttribute = winners.get(n);
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

		double[][] norm = this.normalizationScaling();
		this.translation = norm[0];
		this.scaling = norm[1];

		List<Pair<List<Object>, Double>> distances = new ArrayList<>();

		// determineManhattanDistance
        for (List<Object> trainInstance : this.traindata)
            if (!data.equals(trainInstance))
        		if (this.getMetric() == 0)
					distances.add(new Pair<>(trainInstance, this.determineManhattanDistance(trainInstance, data)));
        		else
					distances.add(new Pair<>(trainInstance, this.determineEuclideanDistance(trainInstance, data)));

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

//		return distances;

		List<Pair<List<Object>, Double>> result = new ArrayList<>();
        while (result.size() < this.getkNearest()) {
            // get sublist of instances with min distance
            List<Pair<List<Object>, Double>> subList = minDistances(distances);
            if (subList.size() + result.size() <= this.getkNearest()) {
                result.addAll(subList);
                distances.removeAll(subList);
            } else {
                Pair<List<Object>, Double> element = randomElement(subList);
                result.add(element);
                distances.remove(element);
            }
        }

		return  result;
	}

    private List<Pair<List<Object>,Double>> minDistances(List<Pair<List<Object>, Double>> distances) {
        List<Pair<List<Object>, Double>> subList = new LinkedList<>();
        double minDistance = distances.get(0).getB();
        for (Pair<List<Object>, Double> pair : distances) {
            if (pair.getB() <= minDistance)
                subList.add(pair);
        }

        return subList;
    }

    private Pair<List<Object>,Double> randomElement(List<Pair<List<Object>, Double>> subList) {
        Random rand = new Random();
        int n = rand.nextInt(subList.size());
        return subList.get(n);
    }

    @Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {

	    double distance = 0.0;

        List<Object> trainInstance = new ArrayList<>();
        List<Object> testInstance = new ArrayList<>();

        // remove class attribute from train & test instance
        if (instance1.size() > instance2.size()) {

			trainInstance.addAll(instance1);
			testInstance.addAll(instance2);

			trainInstance.remove(this.getClassAttribute());
		} else if (instance1.size() < instance2.size()) {

			trainInstance.addAll(instance2);
			testInstance.addAll(instance1);

			trainInstance.remove(this.getClassAttribute());
		} else {
        	trainInstance.addAll(instance1);
			testInstance.addAll(instance2);

			trainInstance.remove(this.getClassAttribute());
			testInstance.remove(this.getClassAttribute());
		}

        for (int i = 0; i < trainInstance.size(); i++) {
            Object trainAttribute = trainInstance.get(i);
            Object testAttribute = testInstance.get(i);

            // if symbolic attribute, use 0/1 distance, otherwise abs(v1 -v2)
            if (trainAttribute instanceof String) {
                if (!trainAttribute.equals(testAttribute))
                    distance += 1;
            } else {
            	if (this.isNormalizing()) {
            	    if (i < this.getClassAttribute()) {
                        trainAttribute = ((Double) trainAttribute - translation[i]) / scaling[i];
                        testAttribute = ((Double) testAttribute - translation[i]) / scaling[i];
                    } else {
                        trainAttribute = ((Double) trainAttribute - translation[i+1]) / scaling[i+1];
                        testAttribute = ((Double) testAttribute - translation[i+1]) / scaling[i+1];
                    }
                }

                distance += Math.abs((Double)trainAttribute - (Double)testAttribute);
            }
        }

		return distance;
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {

		double distance = 0.0;

		List<Object> trainInstance = new ArrayList<>();
		List<Object> testInstance = new ArrayList<>();

		// remove class attribute from train & test instance
		if (instance1.size() > instance2.size()) {

			trainInstance.addAll(instance1);
			testInstance.addAll(instance2);

			trainInstance.remove(this.getClassAttribute());
		} else if (instance1.size() < instance2.size()) {

			trainInstance.addAll(instance2);
			testInstance.addAll(instance1);

			trainInstance.remove(this.getClassAttribute());
		} else {
			trainInstance.addAll(instance1);
			testInstance.addAll(instance2);

			trainInstance.remove(this.getClassAttribute());
			testInstance.remove(this.getClassAttribute());
		}

		for (int i = 0; i < trainInstance.size(); i++) {
			Object trainAttribute = trainInstance.get(i);
			Object testAttribute = testInstance.get(i);

			// if symbolic attribute, use 0/1 distance, otherwise abs(v1 -v2)
			if (trainAttribute instanceof Double) {
                if (this.isNormalizing()) {
                    if (i < this.getClassAttribute()) {
                        trainAttribute = ((Double) trainAttribute - translation[i]) / scaling[i];
                        testAttribute = ((Double) testAttribute - translation[i]) / scaling[i];
                    } else {
                        trainAttribute = ((Double) trainAttribute - translation[i+1]) / scaling[i+1];
                        testAttribute = ((Double) testAttribute - translation[i+1]) / scaling[i+1];
                    }
                }
                distance += Math.abs(Math.pow((Double)trainAttribute - (Double)testAttribute, 2));
			} else {
                if (!trainAttribute.equals(testAttribute))
                    distance += 1;
			}
		}

		distance = Math.sqrt(distance);

		return distance;
	}

	@Override
	protected double[][] normalizationScaling() {
		int amountOfAttributes = this.traindata.get(0).size();
		// initialize arrays for translation(arrays[0]) & scaling(arrays[1])
		double[][] arrays = new double[2][amountOfAttributes];

		// modify arrays if isNormalizing() is true, otherwise return arrays fill with zeros
		if (this.isNormalizing()) {

		    // each attribute has a translation & scaling, corresponding to a column of arrays
			for (int i = 0; i < amountOfAttributes; i++) {
			    // record max and min values, prepare to calculation translation & scaling
				double max = - Double.MAX_VALUE;
				double min = Double.MAX_VALUE;

				for (List<Object> instance : this.traindata) {
					Object attribute = instance.get(i);

					if (attribute instanceof Double) {
						max = ((Double) attribute).compareTo(max) > 0 ? (double) attribute : max;
						min = ((Double) attribute).compareTo(min) < 0 ? (double) attribute : min;
					} else {
					    // attribute is String
						max = 0;
						min = 0;
					}
				}

				arrays[0][i] = min; // translation
                arrays[1][i] = max - min; // scaling

                // sometimes attribute has type double, but only 1 value, so max equals min
                // but this leads to a scaling of 0, which causes error, so manually change it to 1
                if (max == min)
                    arrays[1][i] = 1;

			}
		}

		return arrays;
	}
}
