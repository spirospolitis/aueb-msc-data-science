# AUEB M.Sc. in Data Science
## Course: Data Mining
## Autumn 2019
## Lecturer: I. Kotidis
## Homework: 2
## Author: Manolis Proimakis (https://github.com/manosprom), Spiros Politis

----------

# Cypher queries to Neo4J

For the purpose of the assignment, we have created a Neo4J graph with version 4.0.0. Also the *APOC* plugin should be installed to the Neo4J graph.

Note that the following line should be added to the *neo4j.conf* file, so as to enable JSON importing with the APOC plugin:

```bash
apoc.import.file.enabled=true
```

All Cypher queries have been executed through the Neo4J Browser app.

# Python interface to Neo4J

The current release of Neo4J Python driver (1.7.6) requires a Neo4J graph version 3.5.14. In order to meet the requirements, we have created a graph in Neo4J Desktop that adheres to the required version. Also the *APOC* plugin should be installed to the Neo4J graph.

Note that the following line should be added to the *neo4j.conf* file, so as to enable JSON importing with the APOC plugin:

```bash
apoc.import.file.enabled=true
```

## 1. Python environment setup

This setup requires that Anaconda has been installed on the target machine.

### 1.1. Create Conda virtual env

```
conda create -n msc-ds-elec-dm-homework-2 python=3.7.1
source activate msc-ds-elec-dm-homework-2
```

###  1.2. Install required packages
```
pip install -r requirements.txt
```

###  1.3. Run the Jupyter Notebook

From the command line, cd into the homework directory and type:

```
jupyter notebook
```