# AUEB M.Sc. in Data Science
## Course: Large Scale Data Management
## Title: Report of homework
## Author: Spiros Politis (p3351814)

---

# Asignment instructions

- Download and move shakespeare.txt to hadoop:

        wget https://s3.amazonaws.com/dl4j-distribution/pg100.txt

        mv pg100.txt shakespeare.txt

        hdfs dfs -mkdir /data
        
        hdfs dfs -put shakespeare.txt /data/

---

- Part 1 (0.7 pts). Count all the couples of words Shakespeare ever wrote one next to another in the same line with python and mapReduce.

- Part 2 (0.3 pts). Count all the couples of words Shakespeare ever wrote one next to another in the same line with python and Spark.

- Part 3 (0.7 pts). Count all the couples of words Shakespeare ever wrote in the same line with python and mapReduce.

- Part 4 (0.3 pts). Count all the couples of words Shakespeare ever wrote in the same line with python and Spark.

- Part 5 (0.7 pts). Count all the triplets of words Shakespeare ever wrote in the same line with python and mapReduce.

- Part 6 (0.3 pts). Count all the triplets of words Shakespeare ever wrote in the same line with python and Spark.

Submit the code of your programs, the logs that mapReduce and Spark produced running your code and the first 100 lines of each result file.

---



# Resource preparation

## Hadoop cluster

For implementing the map-reduce part of the homework, we shall use a local cluster spawn with Docker and Docker Compose, as described here: http://www.googlinux.com/quickly-deploy-hadoop-cluster-using-docker/index.html.

The cluster comprises a Hadoop name node, a Yarn master and 3 data nodes.

The process is descibed in a nutshell below:

- Pull required Docker images:

```bash
docker pull swapnillinux/cloudera-hadoop-namenode
docker pull swapnillinux/cloudera-hadoop-yarnmaster
docker pull swapnillinux/cloudera-hadoop-datanode
```

- Create a Docker network:

```bash
docker network hadoop-network
```

- Create persistence volumes for each cluster node:

```bash
docker volume create hadoop-yarnmaster-volume
docker volume create hadoop-namenode-volume
docker volume create hadoop-datanode-1-volume
docker volume create hadoop-datanode-2-volume
docker volume create hadoop-datanode-3-volume
```

- CD into /setup and start the Hadoop cluster with:

```bash
docker-compose up
```

- Access the cluster web console by pointing your browser to http://localhost:8088.



## Spark cluster

For implementing the Spark part of the homework, we shall use a local Spark spawn with Docker and Docker Compose, as described here: https://github.com/dimajix/docker-jupyter-spark.

The cluster comprises a Jupyter notebook server, a Spark master and two Spark slave nodes.

Note that the Hadoop cluster needs to be up and running so as to have access to HDFS.

The Spark cluster created comes with Jupyter Notebook infrastructure installed. In order to access the notebooks server, point your browser to http://localhost:8888.

To start the cluster, follow the procedure outlined below:

- Create a Docker volume for the node in which the Jupyter notebook server resides:

```bash
docker network create spark-jupyter-notebook-volume
```

- Start the cluster by CD-ing into /setup and running:

```bash
docker-compose up
```

- Access the cluster web console by pointing your browser to http://localhost:9090.

---



# Execution and results

## Map-Reduce

- Use the cluster through the namenode:

```bash
docker exec -it namenode bash
```

- Create the the following directory structure on the namenode:

        /
        |
        - homework
               |
            part-1
               |
            part-2
               |
            part-3

- CD into 'homework' directory.

- Retrieve the file 'hadoop-streaming-2.6.0.jar':

```bash
wget http://repo1.maven.org/maven2/org/apache/hadoop/hadoop-streaming/2.6.0/hadoop-streaming-2.6.0.jar
```

- Execute homework instructions to retrieve Shakespeare text and place it on Hadoop.

- Copy homework files (*`run.sh`*, *`mapper.py`*, *`reducer.py`*) for each part into the relevant directory.

- For each part of the homework, a *`run.sh`* script has been created so as to:

  * Delete the corresponding HDFS output path, if it already exists, so that the map-reduce job does not fail when trying to write the output to HDFS.

  * Run the map-reduce job.

- Run each part of the homework by CD-ing into the relevant directory and executing the *`run.sh`* file.

  A log of the *`run.sh`* execution, copied directly from the terminal, has been provided in the corresponding *`job-run.txt`* of each homework part.

- Upon completion of the job, the results are retrieved in a file *`job-output.txt`* by issung the following command:

```bash
hdfs dfs -cat /data/<part-x>/part-00000 > job-output.txt
```

  where *`<part-x>`* is the homework part. 

- To get the top-100 rows, as per homework requirements, we issue the following commands:

```bash
head -100 job-output.txt > job-output-1.txt
rm -y job-output.txt
mv job-output-1.txt job-output.txt
```

Note: in case the map-reduce job fails with "Hadoop Yarn Container Does Not Allocate Enough Space", the remedy is to edit the /etc/hadoop/conf.empty/mapred-site.xml on the namenode and append the following directives:

```bash
<property>
        <name>mapreduce.map.memory.mb</name>
        <value>4096</value>
</property>
<property>
        <name>mapreduce.reduce.memory.mb</name>
        <value>4096</value>
</property>
```

The issue and aforementioned remedy is described here: https://stackoverflow.com/questions/20803577/hadoop-yarn-container-does-not-allocate-enough-space/



## Spark

Each part of the homework, related to Spark, has been implemented as a separate Jupyter notebook. The output of each part of the homework is placed under '/data/*`<part-x>`*/spark', where *`<part-x>`* is the relevant part of the homework. As for the MapReduce part of the homework, the results are retrieved by executing the following steps:

- CD into /homework/*`<part-x>`*/spark, where *`<part-x>`* is the homework part. Notice the *spark* sub-directory.

- The results are retrieved in a file *`job-output.txt`* by issung the following command:

```bash
hdfs dfs -cat /data/<part-x>/spark/part-00000 > job-output.txt
```

  where *`<part-x>`* is the homework part. 

- To get the top-100 rows, as per homework requirements, we issue the following commands:

```bash
head -100 job-output.txt > job-output-1.txt
rm -y job-output.txt
mv job-output-1.txt job-output.txt
```