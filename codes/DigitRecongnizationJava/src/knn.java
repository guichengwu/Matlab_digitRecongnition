import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

public class knn {
	
	public static List<String> knnLabelregcongization(String trainfile, String testfile, boolean header, int k) {
		List<String> labelList = new ArrayList<String>();
		List<List<Double>> trainLists = readCSVFile(trainfile, header);
    	List<List<Double>> testLists = readCSVFile(testfile, header);
    	
		List<List<Integer>> neighborsLists = knnNeighborLists(trainLists, testLists, k);

	    Map<Integer, String> rowLabelMap = buildRowLabelMap(trainLists);
	    
	    labelList = labelDigit(neighborsLists, rowLabelMap);
	    
		return labelList;
	}
	
	public static List<String> labelDigit(List<List<Integer>> neighborsLists, Map<Integer, String> rowLabelMap) {
		List<String> labelList = new ArrayList<String>();
		
		
		
		for (int i = 0; i < neighborsLists.size(); i++) {
			List<Integer> rowsList = neighborsLists.get(i);
			
			//init statistic map 
			Map<String, Integer> statMap = new HashMap<String, Integer>();
			for (int label = 0; label < 10; label++) {
				statMap.put(Integer.toString(label), 0);
			}
			
			for (int j = 0; j < rowsList.size(); j++) {
				String label = rowLabelMap.get(rowsList.get(j));
				statMap.put(label, statMap.get(label)+1);
			}
			
			//sort map
			statMap = MapUtil.sortByValue(statMap);
			//the last entry is the knn label
			String knnLabel = "";
			Iterator<String> iterator = statMap.keySet().iterator();
			while (iterator.hasNext()) {
				knnLabel = iterator.next();
			}
			
			labelList.add(knnLabel);
		}
		
		return labelList;
	}
	
	public static Map<Integer, String> buildRowLabelMap(List<List<Double>> trainLists) {
		
		//Key is the row number, value is the label
		Map<Integer, String> rowLabelMap = new HashMap<Integer, String>();
		
		for (int i = 0; i < trainLists.size(); i++) {
			rowLabelMap.put(i, Integer.toString(trainLists.get(i).get(16).intValue()));
		}
		
		return rowLabelMap;
	}
	public static List<List<Integer>> knnNeighborLists(List<List<Double>> trainLists, List<List<Double>> testLists, int k) {	
		if (testLists == null) {
			return null;
		}
		
		//k nearest neighbors row lists
		List<List<Integer>> neighborsLists = new ArrayList<List<Integer>>();
		
		for (int testRow = 0; testRow < testLists.size(); testRow++) {
			//get each test sample
			List<Double> testList = testLists.get(testRow);
			
			Map<Integer, Double> distanceMap = computeOneSample(testList, trainLists);
            Set<Integer> keySet = distanceMap.keySet();
            Iterator<Integer> iterator = keySet.iterator();
			List<Integer> neighborsList = new ArrayList<Integer>();
			//find the k nearest neighbor
			int tempK = 0;
			while (iterator.hasNext()) {
				neighborsList.add(iterator.next());
				tempK++;
				if (tempK == k) {
					break;
				}
			}
			
			neighborsLists.add(neighborsList);
		}
		
		return neighborsLists;
	}
	
	public static double computeDistance(List<Double> trainList, List<Double> testList) {
		int column = testList.size();
		double distance = 0;
		for (int i = 0; i < column; i++) {
			distance = distance + Math.pow((trainList.get(i) - testList.get(i)), 2);
		}
		
		return Math.sqrt(distance);
	}
	
	public static Map<Integer, Double> computeOneSample(List<Double> testList, List<List<Double>> trainLists) {
		//Key is row number ,and value is distance values
		Map<Integer, Double> distanceMap = new HashMap<Integer, Double>();
		
		for (int i = 0; i < trainLists.size(); i++) {
			List<Double> trainList = trainLists.get(i);
			double distance = computeDistance(trainList, testList);
			distanceMap.put(i, distance);
		}
		//sort map according to distance values
		distanceMap = MapUtil.sortByValue(distanceMap);
		
		return distanceMap;
		
	}
	
	public static List<List<Double>> readCSVFile(String filepath, boolean header) {
		List<List<Double>> rowLists = new ArrayList<List<Double>>();
		try {
			Scanner scanner = new Scanner(new File(filepath));
			//skip the header line
			if (header == true) {
				scanner.nextLine();
			}
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				List<Double> rowList = splitLine(line, ',');
				rowLists.add(rowList);
			}
			
			scanner.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		return rowLists;
	}
	
	public static List<Double> splitLine(String line, char delimiter) {
		String[] tokens = line.split(Character.toString(delimiter));
		List<Double> list = new ArrayList<Double>();
		
		for(int i = 0; i < tokens.length; i++) {
			double d = Double.parseDouble(tokens[i]);
			list.add(d);
		}
		
		return list;
	}
	
	public void testReadCSVFile() {
    	String filepath = "/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-train.csv";
    	boolean header = false;
    	List<List<Double>> rowLists = readCSVFile(filepath, header);
    	
    	StringBuilder builder = new StringBuilder();
    	for (int i = 0; i < rowLists.size(); i++) {
    		List<Double> row = rowLists.get(i);
    		
    		for (int j = 0; j < row.size(); j++) {
    			builder.append(row.get(j).toString());
    			builder.append(' ');
    		}
    		builder.append('\n');
    	}

    	System.out.println(builder.toString());
    	System.out.println("there are " + rowLists.size() + "lines.");
	}
	
	public static void testComputeDistance() {
        Random random = new Random(System.currentTimeMillis());
        List<Double> trainList = new ArrayList<Double>();
        List<Double> testList = new ArrayList<Double>();
        
        for (int i = 0; i < 16; i++) {
        	double d = i;
        	trainList.add(d);
        	System.out.print(d + " ");
        }
        System.out.print('\n');
        
        for (int i = 1; i < 17; i++) {
        	double d = i;
        	testList.add(d);
        	System.out.print(d + " ");
        }
        System.out.print('\n');

        
        System.out.println("distance is: " + computeDistance(trainList, testList));
	}
	
	public static void testComputeOneSample() {
    	String trainfilepath = "/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-train.csv";
    	String testfilepath = "/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-test-nolabels.csv";
    	boolean header = false;
    	List<List<Double>> trainLists = readCSVFile(trainfilepath, header);
    	List<List<Double>> testLists = readCSVFile(testfilepath, header);
    	
    	Map<Integer, Double> distanceMap = computeOneSample(testLists.get(0), trainLists);
    	Iterator iterator = distanceMap.entrySet().iterator();
    	
    	while (iterator.hasNext()) {
    	    Map.Entry<Integer, Double> entry = (Map.Entry<Integer, Double>)iterator.next();
    	    System.out.println("row number: " + entry.getKey() + ", distance: " + entry.getValue());
    	}
	}
	
	public static void testKnnNeighborLists() {
    	int k = 4;
    	String trainfilepath = "/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-train.csv";
    	String testfilepath = "/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-test-nolabels.csv";
    	boolean header = false;
    	List<List<Double>> trainLists = readCSVFile(trainfilepath, header);
    	List<List<Double>> testLists = readCSVFile(testfilepath, header);
    	
    	List<List<Integer>> rowLists = knnNeighborLists(trainLists, testLists, 4);
    	
    	for (int i = 0 ; i < rowLists.size(); i++) {
    		Iterator<Integer> iterator = rowLists.get(i).iterator();
    		
    		while (iterator.hasNext()) {
    			System.out.print(iterator.next() + " ");
    		}
    		System.out.print('\n');
    	}
	}
	
	
	public static void testBuildRowLabelMap() {
		String filepath = "/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-train.csv";
    	boolean header = false;
    	List<List<Double>> trainLists = readCSVFile(filepath, header);
    	
    	Map<Integer, String> rowLabelMap = buildRowLabelMap(trainLists);
    	
    	Iterator iterator = rowLabelMap.entrySet().iterator();
    	while (iterator.hasNext()) {
    		Map.Entry<Integer, String> entry = (Map.Entry<Integer, String>)iterator.next();
    		System.out.println("row number is: " +entry.getKey() + "label is: " + entry.getValue());
    	}
	}
	
	public static void testKnnLabelregcongization(String trainfilepath, String testfilepath, 
			boolean header, int k) {

		List<String> labels = knnLabelregcongization(trainfilepath, testfilepath, header, k);
		
		for (int i = 0; i < labels.size(); i++) {
			System.out.println(labels.get(i));
		}
	
		System.out.println("there are " + labels.size() + "labels");
		
	}
	
	public static void writeCSVFile(List<String> labels, String resultfilePath) {
		try {
			FileWriter writer = new FileWriter(resultfilePath);
			for (int i = 0; i < labels.size(); i++) {
				writer.append(labels.get(i));
				writer.append('\n');
			}
			
			writer.flush();
			writer.close();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public static boolean compareJavaMatlabResults(String javaResultFile, String matlabResultFile) {
		try {
			Scanner javaScanner = new Scanner(new File(javaResultFile));
			Scanner matlabScanner = new Scanner(new File(matlabResultFile));
			int rowIndex = 1;
			int differentNumber = 0;
			while (javaScanner.hasNextLine() || matlabScanner.hasNextLine()) {
				Integer javaResult = Integer.parseInt(javaScanner.nextLine());
			    Integer matlabResult = Integer.parseInt(matlabScanner.nextLine());
				
				if (javaResult != matlabResult) {
					System.out.println(javaResult);
					System.out.println(matlabResult);
					System.out.println("The different row is: " + rowIndex);
					differentNumber++;
				}
				
				rowIndex++;
			}
			javaScanner.close();
			matlabScanner.close();
			
			System.out.println("different numbers : " + differentNumber);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		return true;
	}
	
    public static void main(String[] args) {
    	int k = 4;
    	String trainfilepath = "/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-train.csv";
    	String testfilepath = "/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-test-nolabels.csv";
    	boolean header = false;
    	
    	List<String> labels = knnLabelregcongization(trainfilepath, testfilepath, header, k);
    	
    	String outputFilePath = "/Users/guichengwu/Desktop/ecs 271 assignment 1/javaKNNTestResult.csv";
    	
    	writeCSVFile(labels, outputFilePath);
    	
    	String matlabResultPath = "/Users/guichengwu/Desktop/ecs 271 assignment 1/matlabKNNResult.csv";
    	System.out.println(compareJavaMatlabResults(outputFilePath, matlabResultPath));
    	
    }
}
