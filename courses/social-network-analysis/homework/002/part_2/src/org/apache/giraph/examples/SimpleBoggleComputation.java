/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.giraph.examples;

import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.graph.Vertex;
import org.apache.giraph.examples.SimpleBoggleComputation.TextArrayListWritable;
import org.apache.giraph.utils.ArrayListWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Demonstrates the basic Pregel dispersion implementation.
 */

/**
 * Exercise board:
 * 
 * G   I   Z
 * U   E   K
 * Q   S   E
 * 
 */

@Algorithm(name = "Boggle", description = "Sets the vertex value of each vertex to a list of boggle words ending in that vertex")
public class SimpleBoggleComputation
		extends BasicComputation<Text, TextArrayListWritable, NullWritable, TextArrayListWritable> {

	private static final TreeSet<String> dictionary = new TreeSet<String>(
			Arrays.asList("GEEKS", "SOCIAL", "NETWORK", "ANALYSIS", "QUIZ"));

	// Get an instance of the logger.
	private static final Logger log = Logger.getLogger(SimpleBoggleComputation.class);

	@Override
	public void compute(Vertex<Text, TextArrayListWritable, NullWritable> vertex,
			Iterable<TextArrayListWritable> messages) throws IOException {

		// Every vertex should send an initial message to all its neighbors with
		// a word starting with its respective letter, and the vertex id (so
		// that the same vertex is not used again in this word).
		if (getSuperstep() == 0) {
			TextArrayListWritable talw = new TextArrayListWritable();
			talw.add(new Text(vertex.getId().toString().substring(0, 1)));
			talw.add(vertex.getId());

			log.info("compute::superstep: " + getSuperstep() + " - " + "initial message to all edges: " + "vertex text: " + "'"
					+ talw.get(0).toString() + "'" + ", vertex id: " + "'" + talw.get(1).toString() + "'");

			sendMessageToAllEdges(vertex, talw);
		}

		// For every message a vertex receives, it should add its letter to the
		// word of the message and examine if the word is in the dictionary.
		else {
			// Read messages.
			for (TextArrayListWritable message : messages) {
				// Get message text.
				Text messageText = new Text(message.get(0).toString());

				log.info("compute::superstep: " + getSuperstep() + " - " + "message text: " + "'" + messageText + "'");

				// Get vertex current text.
				Text vertexText = new Text(vertex.getId().toString().substring(0, 1));

				log.info("compute::superstep: " + getSuperstep() + " - " + "vertex text: " + "'" + vertexText + "'");

				// Create the new word with concatenation.
				Text concatenatedText = new Text(messageText.toString() + vertexText.toString());

				log.info("compute::superstep: " + getSuperstep() + " - " + "concatenated text: " + "'" + concatenatedText + "'");

				// If the word is in the dictionary, the vertex should add the
				// word to the value of the vertex.
				// Note that the type value of the vertex is defined as a list
				// of Text objects.
				// Examine if the word is contained in the dictionary.
				if (dictionary.contains(concatenatedText.toString())) {

					log.info("compute::superstep: " + getSuperstep() + " - " + "dictionary contains word " + "'"
							+ concatenatedText + "'");

					// Add the word to the value of the vertex.
					vertex.getValue().add(concatenatedText);
				}

				// If the word is a prefix of a word in the dictionary then an updated message 
				// should be sent to all vertices that have not received it previously.
				// Examine if a word in the dictionary starts with the text.
				else {
					// Create a regex pattern that is the concatenated string.
					Pattern pattern = Pattern.compile(concatenatedText.toString());

					for (String dictionaryEntry : dictionary) {
						// Create a regex matcher.
						Matcher matcher = pattern.matcher(dictionaryEntry);

						// Check if the matcher's prefix match with the
						// matcher's pattern.
						if (matcher.lookingAt()) {

							log.info("compute::superstep: " + getSuperstep() + " - " + "word starting with " + "'"
									+ concatenatedText + "'" + " contained in dictionary");

							TextArrayListWritable talw = new TextArrayListWritable();
							talw.add(concatenatedText);
							talw.add(vertex.getId());
							
							// Send message to vertices that have not received it previously.
							HashSet<Text> unvisitedEdges = this.getUnvisitedEdges(vertex, message);
							sendMessageToMultipleEdges(unvisitedEdges.iterator(), talw);
						}
					}
				}
			}
		}

		vertex.voteToHalt();
	}
	
	/*
	 	Helper method to retrieve a set of unvisited edges of a vertex.
	*/
	private HashSet<Text> getUnvisitedEdges(Vertex<Text, TextArrayListWritable, NullWritable> vertex,
			TextArrayListWritable message) {
		HashSet<Text> visitedEdges = new HashSet<Text>(message.subList(1, message.size()));
		HashSet<Text> allEdges = new HashSet<Text>();
		
		log.info("getUnvisitedEdges::vertex: " + vertex);
		log.info("getUnvisitedEdges::message: " + message);
		
		// Get an iterator of the read-only view of the out-edges of the vertex.
		Iterator<Edge<Text, NullWritable>> edges = vertex.getEdges().iterator();
		
		// Populate allEdges HashSet. 
		while (edges.hasNext()) {
			Edge<Text, NullWritable> edge = edges.next();
			
			log.info("getUnvisitedEdges::edge: " + edge);
			log.info("getUnvisitedEdges::target_vertex_id: " + new Text(edge.getTargetVertexId()));
			
			allEdges.add(new Text(edge.getTargetVertexId()));
		}
		
		// By performing set subtraction, we remove the singular set of visited edges
		// from the set of all edges of the node.
		log.info("getUnvisitedEdges::set changed: " + allEdges.removeAll(visitedEdges));
		
		return allEdges;
	}

	/**
	 * Utility class for delivering the array of vertices THIS vertex should
	 * connect with to close triangles with neighbors
	 */
	public static class TextArrayListWritable extends ArrayListWritable<Text> {
		private static final long serialVersionUID = -7220517688447798587L;

		/** Default constructor for reflection */
		public TextArrayListWritable() {
			super();
		}

		/** Set storage type for this ArrayListWritable */
		@Override
		@SuppressWarnings("unchecked")
		public void setClass() {
			setClass(Text.class);
		}
	}
}
