import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Random;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;

public class DSKMeans {
	private final static String CENTROIDS_FILE_PATH = "input/centroids.txt";
	private final static String TEMP_FILE_PATH = "temp/temp_centroids.txt";
	private final static String FINAL_OUTPUT_FILE_PATH = "output/final_centroids.txt";

	private static String outPath = "output/";
	private static String tempPath = "temp/";

	private final static double epsoln = 0.001;

	private final static int n = 1024; // Number of features

	public static class KMeansMapper extends MapReduceBase implements
			Mapper<LongWritable, Text, Text, Text> {

		@Override
		public void map(LongWritable key, Text val,
				OutputCollector<Text, Text> output, Reporter reporter)
				throws IOException {
			Hashtable<String, ArrayList<Float>> centroids = readCentroids(CENTROIDS_FILE_PATH);

			StringTokenizer itr = new StringTokenizer(val.toString());

			int i = 0;
			ArrayList<Float> recordData = new ArrayList<Float>();
			String token;
			while (itr.hasMoreTokens()) {
				// First two records are the image_url and lush string
				token = itr.nextToken();
				if (i > 1) {
					if (token.equals("MISSING"))
						return;
					recordData.add(Float.parseFloat(token));
				}

				i++;
			}
			String newCentroid = getNearestCentroid(recordData, centroids);
			output.collect(new Text(newCentroid), val);
		}

		private Hashtable<String, ArrayList<Float>> readCentroids(
				String centroidsFilePath) throws IOException {

			Hashtable<String, ArrayList<Float>> centroids = new Hashtable<String, ArrayList<Float>>();

			FileSystem fs = FileSystem.get(new Configuration());
			BufferedReader br = new BufferedReader(new InputStreamReader(
					fs.open(new Path(centroidsFilePath))));

			String line;
			String[] temp;

			while ((line = br.readLine()) != null) {
				temp = line.split(" ");

				ArrayList<Float> features = new ArrayList<Float>();
				// first element is character Ci
				for (int i = 1; i < temp.length; i++)
					features.add(Float.parseFloat(temp[i]));

				centroids.put(temp[0], features);
			}
			return centroids;
		}
	}

	public static class KMeansReducer extends MapReduceBase implements
			Reducer<Text, Text, Text, Text> {

		@Override
		public void reduce(Text key, Iterator<Text> values,
				OutputCollector<Text, Text> output, Reporter reporter)
				throws IOException {
			float[] newCentroidCoordinats = calculateNewCentroid(values);

			String newCentroids = key.toString();
			for (int i = 0; i < newCentroidCoordinats.length; i++)
				newCentroids += " " + newCentroidCoordinats[i];

			FileSystem fs = FileSystem.get(new Configuration());
			if (!fs.exists(new Path(TEMP_FILE_PATH))) {
				// Temp file not created before, so just create it and write the
				// data
				BufferedWriter out = new BufferedWriter(new PrintWriter(
						new OutputStreamWriter(fs.create(new Path(
								TEMP_FILE_PATH)))));

				out.write(newCentroids + "\n");
				out.close();
			} else {
				// Read file and update the record of the key
				BufferedReader br = new BufferedReader(new InputStreamReader(
						fs.open(new Path(TEMP_FILE_PATH))));

				Hashtable<String, String> records = new Hashtable<String, String>();

				String line;
				String[] temp;
				while ((line = br.readLine()) != null) {
					temp = line.split(" ", 2);
					records.put(temp[0], line);
				}

				if (records.containsKey(key.toString())) {
					records.remove(key.toString());
				}
				records.put(key.toString(), newCentroids);

				BufferedWriter out = new BufferedWriter(new PrintWriter(
						new OutputStreamWriter(fs.create(new Path(
								TEMP_FILE_PATH)))));

				for (String tempKey : records.keySet())
					out.write(records.get(tempKey) + "\n");

				out.close();
			}
		}

		private float[] calculateNewCentroid(Iterator<Text> values) {
			float[] newCoordinates = new float[n];

			int recordCount = 0;
			while (values.hasNext()) {
				recordCount++;

				StringTokenizer itr = new StringTokenizer(values.next()
						.toString());

				int i = 0;
				String token;
				while (itr.hasMoreTokens()) {
					token = itr.nextToken();
					if (i > 1)
						newCoordinates[i] += Float.parseFloat(token);

					i++;
				}
			}

			for (int i = 0; i < newCoordinates.length; i++) {
				newCoordinates[i] = (float) (newCoordinates[i] / recordCount * 1.0);
			}
			return newCoordinates;
		}
	}

	/**
	 * 
	 * @param args
	 *            args[0] = input path
	 * 
	 *            args[1] = output path
	 * 
	 *            args[2] = Number of clusters "K"
	 */
	public static void main(String[] args) {
		// Set configuration of the Job
		JobConf conf = new JobConf(DSKMeans.class);
		conf.setJobName("KMeans");

		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(Text.class);

		TextInputFormat.addInputPath(conf, new Path(args[0]));
		TextOutputFormat.setOutputPath(conf, new Path(args[1]));

		outPath = args[1];

		// Set Mapper and Reducer
		conf.setMapperClass(KMeansMapper.class);
		conf.setReducerClass(KMeansReducer.class);

		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(TextOutputFormat.class);

		try {
			int k = Integer.parseInt(args[2]);
			generateRandomCentroids(k, n, FileSystem.get(conf));

			FileSystem.get(conf).delete(new Path(outPath), true);
			FileSystem.get(conf).delete(new Path(TEMP_FILE_PATH), true);
			
			do {
				JobClient jobClient = new JobClient();
				jobClient.setConf(conf);
				JobClient.runJob(conf);
			} while (!isDone(FileSystem.get(conf)));

		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

	private static void generateRandomCentroids(int k, int n, FileSystem fs)
			throws IOException {
		BufferedWriter out = new BufferedWriter(
				new PrintWriter(new OutputStreamWriter(fs.create(new Path(
						CENTROIDS_FILE_PATH)))));

		Random r = new Random();
		for (int i = 1; i <= k; i++) {
			out.write("C" + i);
			for (int j = 0; j < n; j++) {
				out.write(" " + r.nextFloat());
			}
			out.write("\n");
		}

		out.close();
	}

	// Compare between the temporary file and the centroids files and if no
	// changes happened return true otherwise return false.
	private static boolean isDone(FileSystem fs) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(
				fs.open(new Path(CENTROIDS_FILE_PATH))));

		Hashtable<String, String> centroids = new Hashtable<String, String>();

		String line;
		String[] temp;
		while ((line = br.readLine()) != null) {
			temp = line.split(" ", 2);
			centroids.put(temp[0], temp[1]);
		}
		br.close();
		
		br = new BufferedReader(new InputStreamReader(fs.open(new Path(
				TEMP_FILE_PATH))));

		Hashtable<String, String> tempCentroids = new Hashtable<String, String>();

		while ((line = br.readLine()) != null) {
			temp = line.split(" ", 2);
			tempCentroids.put(temp[0], temp[1]);
		}
		br.close();
		
		String[] records;
		String[] tempRecords;
		boolean isDone = true;
		for (String key : centroids.keySet()) {
			if (!tempCentroids.containsKey(key)) {
				isDone = false;
				break;
			}

			records = centroids.get(key).split(" ");
			tempRecords = tempCentroids.get(key).split(" ");

			for (int i = 0; i < tempRecords.length; i++) {
				if (Math.abs(Float.parseFloat(records[i])
						- Float.parseFloat(tempRecords[i])) > epsoln) {
					isDone = false;
					break;
				}
			}
		}

		String filePath = "";
		if (!isDone) {
			// Delete OutputPath
			fs.delete(new Path(outPath), true);
			filePath = CENTROIDS_FILE_PATH;
		} else {
			fs.delete(new Path(tempPath), true);
			filePath = FINAL_OUTPUT_FILE_PATH;
		}

		// Update centroid file
		BufferedWriter out = new BufferedWriter(new PrintWriter(
				new OutputStreamWriter(fs.create(new Path(filePath)))));
		for (String key : tempCentroids.keySet()) {
			out.write(key + " " + tempCentroids.get(key) + "\n");
		}
		out.close();
		return isDone;
	}

	private static String getNearestCentroid(ArrayList<Float> recordData,
			Hashtable<String, ArrayList<Float>> centroids) {

		Hashtable<Float, String> distanceToKeyContainer = new Hashtable<Float, String>();

		int dist = 0;

		for (String key : centroids.keySet()) {
			ArrayList<Float> data = centroids.get(key);
			for (int j = 0; j < recordData.size(); j++) {
				dist += Math.pow(data.get(j) - recordData.get(j), 2);
			}
			distanceToKeyContainer.put((float) Math.sqrt(dist), key);
			dist = 0;
		}

		// Get Min dist
		float min = Float.MAX_VALUE;
		String minCentroidLabel = "";
		for (Float val : distanceToKeyContainer.keySet()) {
			if (val < min) {
				min = val;
				minCentroidLabel = distanceToKeyContainer.get(val);
			}
		}
		System.out.println(minCentroidLabel);
		return minCentroidLabel;
	}
}
