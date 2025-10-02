# sudo neo4j-admin database import full --nodes=amazon/nodes.csv --nodes=goodreads/nodes.csv --nodes=maple/Physics/nodes.csv --relationships=amazon/edges.csv --relationships=goodreads/edges.csv --relationships=maple/Physics/edges.csv --overwrite-destination --verbose

sudo neo4j-admin database import full --nodes=maple/Physics/nodes.csv --relationships=maple/Physics/edges.csv --overwrite-destination --verbose

CREATE INDEX IF NOT EXISTS FOR (n:physics) ON (n.id);
CREATE INDEX IF NOT EXISTS FOR (n:goodreads) ON (n.id);
CREATE INDEX IF NOT EXISTS FOR (n:amazon) ON (n.id);
