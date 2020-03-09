import logging
from neo4j import GraphDatabase, Statement
from neo4j.types.graph import Node
import numpy as np
import pandas as pd

"""
    Implements parts of the Data Mining 2nd assignment.
"""
class Neo4jPy(object):
    """
        Constructor.
    """
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth = (user, password), encrypted=False)

        logging.basicConfig(
            format = "%(levelname)s: %(message)s", 
            level = logging.INFO, 
            datefmt = "%I:%M:%S"
        )
        self.__log = logging.getLogger()
    
    """
        Clean-up after the oneself.
    """
    def close(self):
        self._driver.close()
    
    """
    """            
    def load_json_data(self):
        cypher = """
            CALL apoc.periodic.iterate(
                'UNWIND ["dblp-ref-0.json", "dblp-ref-1.json", "dblp-ref-2.json", "dblp-ref-3.json"] AS file CALL apoc.load.json(file) YIELD value as a return a',
                'MERGE (article:Article {id: a.id}) 
                    ON CREATE SET article.title = a.title, article.year = a.year, article.abstract = a.abstract
                    ON MATCH SET article.title = a.title, article.year = a.year, article.abstract = a.abstract
                FOREACH (authorName in a.authors | 
                    MERGE (author:Author {name: authorName}) 
                    MERGE (author)-[:WROTE {year: article.year}]->(article)) 
                    MERGE (venue:Venue {name: a.venue }) 
                    MERGE (venue)-[:PRESENTED]->(article)
                FOREACH (referenceId in a.references | MERGE (referenceArticle:Article {id: referenceId}) MERGE (article)-[:CITES]->(referenceArticle))',
                {batchSize:100}
                )
            YIELD batches,total return batches, total
        """
        try:
            with self._driver.session() as session:
                session.run(Statement(cypher))

                # Verify data loaded.
                self.__log.info("Loaded {} articles".format(session.run("MATCH(n:Article) return count(n)").single()[0]))
                self.__log.info("Loaded {} authors".format(session.run("MATCH(n:Author) return count(n)").single()[0]))
                self.__log.info("Loaded {} venues".format(session.run("MATCH(n:Venue) return count(n)").single()[0]))
                self.__log.info("Loaded {} citations".format(session.run("MATCH ()-[r:CITES]->() RETURN count(*) as count").single()[0]))
        except Exception as e:
            self.__log.error("{}".format(e))
    
    """
    """
    def execute_statement(self, cypher: str):
        try:
            with self._driver.session() as session:
                result = session.run(Statement(cypher))
        except Exception as e:
            self.__log.error("{}".format(e))
    
    """
    """
    def query(self, cypher: str):
        try:
            with self._driver.session() as session:
                result = session.run(Statement(cypher))
                
            return [record for record in result]
        except Exception as e:
            self.__log.error("{}".format(e))
            
            return None
    
    """
        Returns a Bolt resultset as a Pandas DataFrame.
        
        :param result: the Bolt resultset.
        
        :return: Pandas DataFrame.
    """
    def to_df(self, result):
        record_keys = result[0].keys()
        
        data = np.zeros((len(result), len(result[0].keys())), dtype = object)
            
        for i, record in enumerate(result):
            
            column_index = 0
            
            for record_key in record_keys:
                # Graph node.
                if isinstance(record[record_key], Node):
                    node = record[record_key]
                    
                    for key, value in node.items():
                        data[i, column_index] = value
                        column_index += 1
                else:
                    data[i, column_index] = record[record_key]

                    column_index += 1
        
        df = pd.DataFrame(
            data = data, 
            columns = result[0].keys()
        )

        return df