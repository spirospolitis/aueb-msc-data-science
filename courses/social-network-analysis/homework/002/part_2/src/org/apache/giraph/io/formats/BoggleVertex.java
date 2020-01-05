package org.apache.giraph.io.formats;

import java.util.Iterator;

import org.apache.giraph.graph.Vertex;
import org.apache.giraph.conf.DefaultImmutableClassesGiraphConfigurable;
import org.apache.giraph.conf.GiraphConfiguration;
import org.apache.giraph.conf.ImmutableClassesGiraphConfiguration;
import org.apache.giraph.edge.ByteArrayEdges;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.edge.MutableEdge;
import org.apache.giraph.edge.MutableEdgesIterable;
import org.apache.giraph.edge.MutableEdgesWrapper;
import org.apache.giraph.edge.MutableOutEdges;
import org.apache.giraph.edge.OutEdges;
import org.apache.giraph.edge.StrictRandomAccessOutEdges;
import org.apache.giraph.examples.SimpleBoggleComputation.TextArrayListWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;

public class BoggleVertex extends DefaultImmutableClassesGiraphConfigurable<Text, TextArrayListWritable, NullWritable>
	implements Vertex<Text, TextArrayListWritable, NullWritable>{

	 /** Vertex id. */
	  private Text id;
	  /** Vertex value. */
	  private TextArrayListWritable value;
	  /** Outgoing edges. */
	  private OutEdges<Text, NullWritable> edges;
	  /** If true, do not do anymore computation on this vertex. */
	  private boolean halt;
	  
	  
	@Override
	public void addEdge(Edge<Text, NullWritable> edge) {
		edges.add(edge);
	}
	
	@Override
	public Iterable<NullWritable> getAllEdgeValues(Text targetVertexId) {
		return null;
	}
	
	@Override
	public NullWritable getEdgeValue(Text targetVertexId) {
		return NullWritable.get();
	}
	
	@Override
	public Iterable<Edge<Text, NullWritable>> getEdges() {
		return edges;
	}
	
	@Override
	public Text getId() {
		return id;
	}
	
	@Override
	public Iterable<MutableEdge<Text, NullWritable>> getMutableEdges() {
		  // If the OutEdges implementation has a specialized mutable iterator,
	    // we use that; otherwise, we build a new data structure as we iterate
	    // over the current edges.
	    if (edges instanceof MutableOutEdges) {
	      return new Iterable<MutableEdge<Text, NullWritable>>() {
	        @Override
	        public Iterator<MutableEdge<Text, NullWritable>> iterator() {
	          return ((MutableOutEdges<Text, NullWritable>) edges).mutableIterator();
	        }
	      };
	    } else {
	      return new MutableEdgesIterable<Text, NullWritable>(this);
	    }
	}
	
	@Override
	public int getNumEdges() {
		return edges.size();
	}
	
	@Override
	public TextArrayListWritable getValue() {
		return value;
	}
	
	@Override
	public void initialize(Text id, TextArrayListWritable value) {
		this.id = id;
	    this.value = value;
	    this.edges = new ByteArrayEdges<Text, NullWritable>();
		
	}
	@Override
	public void initialize(Text id, TextArrayListWritable value,
			Iterable<Edge<Text, NullWritable>> edges) {
		this.id = id;
	    this.value = value;
	    setEdges(edges);
	}
	
	@Override
	public boolean isHalted() {
		return halt;
	}
	
	@Override
	public void removeEdges(Text targetVertexId) {
		edges.remove(targetVertexId);
	}
	
	@Override
	public void setEdgeValue(Text targetVertexId, NullWritable edgeValue) {
		// If the OutEdges implementation has a specialized random-access
	    // method, we use that; otherwise, we scan the edges.
	    if (edges instanceof StrictRandomAccessOutEdges) {
	      ((StrictRandomAccessOutEdges<Text, NullWritable>) edges).setEdgeValue(
	          targetVertexId, edgeValue);
	    } else {
	      for (MutableEdge<Text, NullWritable> edge : getMutableEdges()) {
	        if (edge.getTargetVertexId().equals(targetVertexId)) {
	          edge.setValue(edgeValue);
	        }
	      }
	    }
	}
	
	@Override
	public void setEdges(Iterable<Edge<Text, NullWritable>> edges) {
	    // If the iterable is actually an instance of OutEdges,
	    // we simply take the reference.
	    // Otherwise, we initialize a new OutEdges.
	    if (edges instanceof OutEdges) {
	      this.edges = (OutEdges<Text, NullWritable>) edges;
	    } else {
	    	GiraphConfiguration gc = new GiraphConfiguration();
	    	gc.setOutEdgesClass(ByteArrayEdges.class);
	    	ImmutableClassesGiraphConfiguration<Text, TextArrayListWritable, NullWritable> immutableClassesGiraphConfiguration = new ImmutableClassesGiraphConfiguration<>(gc);
	    	this.edges = immutableClassesGiraphConfiguration.createOutEdges();
	    	this.edges.initialize(edges);
	    }
	}
	
	@Override
	public void setValue(TextArrayListWritable value) {
		this.value = value;
	}
	
	@Override
	public void unwrapMutableEdges() {
		  if (edges instanceof MutableEdgesWrapper) {
		      edges = ((MutableEdgesWrapper<Text, NullWritable>) edges).unwrap();
		    }
	}
	
	@Override
	public void voteToHalt() {
	    halt = true;
	}
	
	@Override
	public void wakeUp() {
	    halt = false;
	}


}
