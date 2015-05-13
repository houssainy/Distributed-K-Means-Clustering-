//import java.io.BufferedReader;
//import java.io.FileReader;
//
//import weka.core.Attribute;
//import weka.core.FastVector;
//import weka.core.Instance;
//import weka.core.Instances;
//import weka.clusterers.SimpleKMeans;
//
//public class KMeans {
//
//	public static void main(String[] args) throws Exception {
//		int n = 1024;
//		SimpleKMeans kmeans = new SimpleKMeans();
//
//		kmeans.setSeed(10);
//
//		// important parameter to set: preserver order, number of cluster.
//		kmeans.setPreserveInstancesOrder(true);
//		kmeans.setNumClusters(Integer.parseInt(args[0]));
//
//		FastVector fvWekaAttributes = new FastVector(n);
//		for (int i = 0; i < n; i++) {
//			Attribute Attribute = new Attribute(i + "Numeric");
//			fvWekaAttributes.addElement(Attribute);
//		}
//
//		// Create an empty training set
//		Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, n);
//
//		BufferedReader br = new BufferedReader(new FileReader("input.txt"));
//		String line;
//		while ((line = br.readLine()) != null) {
//			 = line.split("\\t");
//			String[] parts1 = parts[2].split(" ");
//
//			if (!parts1[0].equals("MISSING")) {
//				Instance iExample = new Instance(n);
//				for (int i = 0; i < parts1.length; i++) {
//					iExample.setValue(
//							(Attribute) fvWekaAttributes.elementAt(i),
//							Double.parseDouble(parts1[i]));
//				}
//				isTrainingSet.add(iExample);
//			}
//		}
//		kmeans.buildClusterer(isTrainingSet);
//		
////		BufferedReader br = new BufferedReader(new FileReader("input.txt"));
////		Instance x = new Instance(n);
////		
////		line = br1.readLine();
////		String[] parts = line.split("\\t");
////		String[] parts1 = parts[2].split(" ");
////		
////		for (int i = 0; i < parts1.length; i++) {
////			x.setValue(
////					(Attribute) fvWekaAttributes.elementAt(i),
////					Double.parseDouble(parts1[i]));
////		}
////		
////		System.out.println(kmeans.clusterInstance(x));
//		
//		int[] assignments = kmeans.getAssignments();
//
//		int i = 0;
//		for (int clusterNum : assignments) {
//			System.out.printf("Instance %d -> Cluster %d \n", i, clusterNum);
//			i++;
//		}
//	}
//}