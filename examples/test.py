from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "123456789"))

# Access driver configuration
print(driver._config.max_connection_pool_size)  # Default: 100
print(driver._config.connection_timeout)       # Default: 30 seconds

driver.close()