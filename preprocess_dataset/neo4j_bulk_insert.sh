sudo neo4j-admin database import full --nodes=amazon/nodes.csv --nodes=goodreads/nodes.csv --nodes=physics/nodes.csv --relationships=amazon/edges.csv --relationships=goodreads/edges.csv --relationships=physics/edges.csv --overwrite-destination --verbose

CREATE INDEX IF NOT EXISTS FOR (n:physics) ON (n.id);
CREATE INDEX IF NOT EXISTS FOR (n:goodreads) ON (n.id);
CREATE INDEX IF NOT EXISTS FOR (n:amazon) ON (n.id);
