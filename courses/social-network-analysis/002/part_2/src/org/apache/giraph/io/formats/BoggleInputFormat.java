package org.apache.giraph.io.formats;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.giraph.graph.Vertex;
import org.apache.giraph.io.formats.TextVertexInputFormat;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.edge.EdgeFactory;
import org.apache.giraph.examples.SimpleBoggleComputation.TextArrayListWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

public class BoggleInputFormat extends TextVertexInputFormat<Text, TextArrayListWritable, NullWritable>{


	@Override
	public TextVertexReader createVertexReader(
			InputSplit split, TaskAttemptContext context) throws IOException {
		return new ComponentisationVertexReader();
	}

	public class ComponentisationVertexReader extends TextVertexReader {

		@Override
		public boolean nextVertex() throws IOException, InterruptedException {
			return getRecordReader().nextKeyValue();
		}
		
		@Override
		public Vertex<Text, TextArrayListWritable, NullWritable> getCurrentVertex() throws IOException, InterruptedException {
			Text line = getRecordReader().getCurrentValue();
			String[] parts = line.toString().split(" ");
			Text id = new Text(parts[0]);
			
			ArrayList<Edge<Text, NullWritable>> edgeIdList = new ArrayList<Edge<Text, NullWritable>>();
			 
			if(parts.length > 1) {
				for (int i = 1; i < parts.length; i++) {
					Edge<Text, NullWritable> edge = EdgeFactory.create(new Text(parts[i]));
					edgeIdList.add(edge);
				}
			}
		    Vertex<Text, TextArrayListWritable, NullWritable> vertex = new BoggleVertex();
		    vertex.initialize(id, new TextArrayListWritable(), edgeIdList);
		    return vertex;
		}

}

}
