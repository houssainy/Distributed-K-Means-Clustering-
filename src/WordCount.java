import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;

public class WordCount {
	public static class LineIndexMapper extends MapReduceBase implements
			Mapper<LongWritable, Text, Text, Text> {

		private final static Text word = new Text();
		private final static Text location = new Text();

		public void map(LongWritable key, Text val,
				OutputCollector<Text, Text> output, Reporter reporter)
				throws IOException {
			System.out.println("I am in mapper....................");
			FileSplit fileSplit = (FileSplit) reporter.getInputSplit();
			String fileName = fileSplit.getPath().getName();
			location.set(fileName);

			String line = val.toString();
			StringTokenizer itr = new StringTokenizer(line.toLowerCase());
			while (itr.hasMoreTokens()) {
				word.set(itr.nextToken());
				output.collect(word, location);
			}
		}
	}

	public static class LineIndexReducer extends MapReduceBase implements
			Reducer<Text, Text, Text, Text> {

		@Override
		public void reduce(Text key, Iterator<Text> values,
				OutputCollector<Text, Text> output, Reporter reporter)
				throws IOException {
			System.out.println("I am in reducer....................");
			boolean first = true;
			StringBuilder toReturn = new StringBuilder();
			while (values.hasNext()) {
				if (!first)
					toReturn.append(", ");
				first = false;
				toReturn.append(values.next().toString());
			}

			output.collect(key, new Text(toReturn.toString()));

			FileSystem fs = FileSystem.get(new Configuration());

			PrintWriter pr = new PrintWriter(new OutputStreamWriter(
					fs.create(new Path("input/houssainy.txt"))));
			pr.write("helloWorld\nhhhhhhhhhhhhhhhhh");
			pr.close();
			BufferedReader br = new BufferedReader(new InputStreamReader(
					fs.open(new Path("input/houssainy.txt"))));

			System.out.println("*********************");
			System.out.println(br.readLine());
		}
	}

	private OutputCollector<String, String> x = new OutputCollector<String, String>() {

		@Override
		public void collect(String arg0, String arg1) throws IOException {
			// TODO Auto-generated method stub

		}
	};

	/**
	 * The actual main() method for our program; this is the "driver" for the
	 * MapReduce job.
	 */
	public static void main(String[] args) {
		JobClient client = new JobClient();
		JobConf conf = new JobConf(WordCount.class);

		conf.setJobName("WordCount");

		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(Text.class);

		FileInputFormat.addInputPath(conf, new Path(args[0]));
		FileOutputFormat.setOutputPath(conf, new Path(args[1]));

		conf.setMapperClass(LineIndexMapper.class);
		conf.setReducerClass(LineIndexReducer.class);

		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(TextOutputFormat.class);

		conf.setNumReduceTasks(2);

		client.setConf(conf);

		try {
			JobClient.runJob(conf);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}