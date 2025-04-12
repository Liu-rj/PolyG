"""
Reference:
 - Some Prompts are from [graphrag](https://github.com/microsoft/graphrag) and [nano-graphrag](https://github.com/gusye1234/nano-graphrag)
"""

PHYSICS_GRAPH_SCHEMA = """
Definition of the graph:
This knowledge graph is a citation graph in physics, there are three types of nodes in this graph: paper, author and venue.

Node properties:
1. type: author, properties: ["id", "name", "node_type"]
2. type: paper, properties: ["id", "name", "label", "year", "node_type", "abstract"]
3. type: venue, properties: ["id", "name", "node_type"]

Edge properties:
Author nodes are linked to their paper nodes by authorship. Specific relations are:
1. author -> "paper" -> paper

Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Specific relations are:
1. paper -> "author" -> author
2. paper -> "reference" -> paper
3. paper -> "venue" -> venue
3. paper -> "cited_by" -> paper

Venue nodes are linked to their included paper nodes. Specific relations are:
1. venue -> "paper" -> paper

Graph indentifier:
Add the identification label "physics" to the entities in the cypher query, for example, ":physics:author" (same for other types of entities).
"""

GOODREADS_GRAPH_SCHEMA = """
Definition of the graph:
This knowledge graph is a literature graph names goodreads, there are four types of nodes in this graph: book, author, publisher and series.

Node properties:
1. type: book, properties: ["id", "name", "node_type", "description", "publication_year", "genres"]
2. type: author, properties: ["id", "name", "node_type"]
3. type: publisher, properties: ["id", "name", "node_type"]
4. type: series, properties: ["id", "name", "node_type", "description"]

Edge properties:
Book nodes are linked to neighboring book nodes, author nodes, publisher nodes and series nodes. Specific relations are:
1. book -> "author" -> author
2. book -> "publisher" -> publisher
3. book -> "series" -> series
4. book -> "similar_books" -> book

Author nodes are linked to their neighboring book nodes. Specific relations are:
1. author -> "book" -> book

Publisher nodes are linked to their neighboring book nodes. Specific relations are:
1. publisher -> "book" -> book

Series nodes are linked to their neighboring book nodes. Specific relations are:
1. series -> "book" -> book

Graph indentifier:
Add the identification label "goodreads" to the entities in the cypher query, for example, ":goodreads:author" (same for other types of entities).
"""

AMAZON_GRAPH_SCHEMA = """
Definition of the graph:
This knowledge graph is an e-commerce graph in amazon, there are two types of nodes in this graph: item and brand.

Node properties:
1. type: item, properties: ["id", "name", "node_type"]
2. type: brand, properties: ["id", "name", "node_type"]

Edge properties:
Item nodes are linked to neighboring item nodes and brand nodes. Specific relations are:
1. item -> "also_viewed_item" -> item
2. item -> "buy_after_viewing_item" -> item
3. item -> "also_bought_item" -> item
4. item -> "bought_together_item" -> item
5. item -> "brand" -> brand

Brand nodes are linked to their neighboring item nodes. Specific relations are:
1. brand -> "item" -> item

Graph indentifier:
Add the identification label "amazon" to the entities in the cypher query, for example, ":amazon:items" (same for other types of entities).
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS[
    "local_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.

---Setting---

The data tables may appear in one of the two forms:
1. Two tables are provided: one being the entity node and one being the edge relationship.
2. Multiple reasoning paths are provided, indicating relations between entities.

Questions can be about node inquries or relations between nodes.

Entities in the question may not have direct relationship and answering the question will need to consider multi-hop relations.

---Goal---

Generate a response to the user's question following the given target length and format, only based on relevant information in the provided data tables.

Users may not see the provided data tables, so provied necessary evidence in your response.

If you don't know the answer, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.

Note: In your reponses, keep the name of entities as they are, do not change them in any way.

---Target response length and format---

{response_type}

---Data tables---

{context_data}
"""

PROMPTS[
    "cypher_answer_summary"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.

---Setting---

The rows in the data table are answer entities to user questions given by cypher queries, please give a summary based on the answers.

For example, if the question is "Who is the the author of the book 'xxx'?", then the author information is already listed as rows in the answer table.

You don't need to come up with the answers yourself as they are already given, just give a summary based on the answers.

Note: Keep the name of answer entities as they are in the answer table, do not change them in any way.

---Goal---

The summary should be comprehensive, diverse and empowerful, which can thoroughly cover diverse aspects, enable the reader to understand the topic and make informed judgments.

---Target response length and format---

{response_type}

---Answer table---

{context_data}
"""

PROMPTS[
    "cypher_query_prompt"
] = """---Role---

You are a helpful assistant that can generate trustful reasoning paths with the knowledge of the graph schema to answer the given user question.

---Setting---

You will be provided with the knowledge graph schema, which indicates the types of nodes and edges, and by what relations nodes are connected.

User questions are about node inquries which involve multi-hop relation paths.

---Goal---

Generate a trustful reasoning path that can be executed on graph using the given user question, based on the given schema of the knowledge graph.

Also give the cypher query that can be executed on the graph to get the answer.

If you don't have adequate information to give trustful reasoning paths, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.

---Graph Schema---

{graph_schema}

---Notes---

1. Please carefully think about the query structure and make sure the query is **correct** and **efficient** to execute. Do not forget to assign a variable name before retrieving the attributes.

2. Add the identification label in the generated cypher query as instructed in the graph schema section to ensure the cypher query inquires about the right graph.

3. Use the "id" property to identify the entities in the graph, rather than names. Users will provide the ids of the entities along with the questions.

4. Always use "->" rather than "<-" and "-" to indicate the direction of the relationship in the cypher query. Each edge have an reversed edge in the graph, so do not involve duplicated paths.

5. Return the unique ids of retrieved entities and set the label of the result column as "id", using `RETURN DISTINCT node.id as id` in the cypher query.

6. Use "LIMIT 20" to limit the number of results returned in the cypher query.

7. return the cypher query in the following format:
```cypher
Your cypher query here
```
"""


PROMPTS[
    "cypher_path_search_prompt_physics"
] = """---Role---

You are a helpful assistant that can generate trustful reasoning paths with the knowledge of the graph schema to answer the given user question.

---Setting---

You will be provided with the knowledge graph schema, which indicates the types of nodes and edges, and by what relations nodes are connected.

User questions are about relationships between nodes which can be indirect and result in multi-hop relation paths. User questions may include a relation constraint that indicates some specific paths.

An example:
User question is "What is the relationship between 'J. Koll' and 'Z. Staykova' in terms of paper reference?". In this case, the user is asking about the paths between two authors that is constraint to paper references and citations.
Suppose the "id" of 'J. Koll' and 'Z. Staykova' is 'id1' and 'id2', and the cypher query for this question is (where only paths that contains reference and citation relations should be considered):
```cypher
MATCH path = (author1:author {id: 'id1'})-[:paper]->(paper1:paper)-[:reference|cited_by]->(paper2:paper)-[:author]->(author2:author {id: 'id2'})
RETURN path
LIMIT 10
```

---Goal---

Generate a trustful reasoning path that can be executed on graph using the given user question, based on the given schema of the knowledge graph. Also give the cypher query that can be executed on the graph to get the answer.

If you don't have adequate information to give trustful reasoning paths, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.

---Graph Schema---

{graph_schema}

---Notes---

1. Add the identification label in the generated cypher query as instructed in the graph schema section to ensure the cypher query inquires about the right graph.

2. Use the "id" property to identify the entities in the graph, rather than names. Users will provide the ids of the entities along with the questions.

3. Please carefully think about the query structure and make sure the query is *correct* and *efficient* to execute.
For example, correct cypher query should first "MATCH path" then "RETURN path".
Also, cypher does not allow mixing label expression symbols ('|', '&', '!', and '%') with colon (':') between labels. To match nodes with any type, just use "(:physics)" is fine.

4. Start from short-path cypher queries and do not use `*1..`, `*1..2`, `*1..3`, `*..` or even larger ranges for elation matching if we don't explicitly tell you to do so as it will take a long time.
For example, just starting from "(:paper)-[:reference|cited_by]->(:paper)" for matching "paper reference" relations is good. If current simple queries can not find the answer, we will ask you to gradually extend the path with specfic path length.

5. Always use "->" rather than "<-" and "-" to indicate the direction of the relationship in the cypher query. Each edge have an reversed edge in the graph, so do not involve duplicated paths.

6. Always use single "MATCH path =" clause for the whole cypher query, starting from one input entity to another and captures the whole path.

7. Return the results in path format:
```cypher
RETURN path
LIMIT 10
```
"""


PROMPTS[
    "cypher_path_search_prompt_goodreads"
] = """---Role---

You are a helpful assistant that can generate trustful reasoning paths with the knowledge of the graph schema to answer the given user question.

---Setting---

You will be provided with the knowledge graph schema, which indicates the types of nodes and edges, and by what relations nodes are connected.

User questions are about relationships between nodes which can be indirect and result in multi-hop relation paths. User questions may include a relation constraint that indicates some specific paths.

An example:
User question is "What is the relationship between authors 'A' and 'B' regarding collaborated books?". In this case, the user is asking about the paths between two authors that is constraint to books they have co-authored.
Suppose the "id"s of author 'A' and 'B' are 'id1' and 'id2', and the cypher query for this question is (where only paths that contains co-authored books should be considered):
```cypher
MATCH path = (author1:author {id: 'id1'})-[:book]->(book1:book)-[:author]->(author2:author {id: 'id2'})
RETURN path
LIMIT 10
```

---Goal---

Generate a trustful reasoning path that can be executed on graph using the given user question, based on the given schema of the knowledge graph. Also give the cypher query that can be executed on the graph to get the answer.

If you don't have adequate information to give trustful reasoning paths, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.

---Graph Schema---

{graph_schema}

---Notes---

1. Add the identification label in the generated cypher query as instructed in the graph schema section to ensure the cypher query inquires about the right graph.

2. Use the "id" property to identify the entities in the graph, rather than names. Users will provide the ids of the entities along with the questions.

3. Please carefully think about the query structure and make sure the query is *correct* and *efficient* to execute.
For example, correct cypher query should first "MATCH path" then "RETURN path".
Also, cypher does not allow mixing label expression symbols ('|', '&', '!', and '%') with colon (':') between labels. To match nodes with any type, just use "(:goodreads)" is fine.

4. Start from short-path cypher queries and do not use `*1..`, `*1..2`, `*1..3`, `*..` or even larger ranges for elation matching if we don't explicitly tell you to do so as it will take a long time.
If current simple queries can not find the answer, we will ask you to gradually extend the path with specfic path length.

5. Always use "->" rather than "<-" and "-" to indicate the direction of the relationship in the cypher query. Each edge have an reversed edge in the graph, so do not involve duplicated paths.

6. Always use single "MATCH path =" clause for the whole cypher query, starting from one input entity to another and captures the whole path.

7. Return the results in path format:
```cypher
RETURN path
LIMIT 10
```
"""


PROMPTS[
    "cypher_path_search_prompt_amazon"
] = """---Role---

You are a helpful assistant that can generate trustful reasoning paths with the knowledge of the graph schema to answer the given user question.

---Setting---

You will be provided with the knowledge graph schema, which indicates the types of nodes and edges, and by what relations nodes are connected.

User questions are about relationships between nodes which can be indirect and result in multi-hop relation paths. User questions may include a relation constraint that indicates some specific paths.

An example:
User question is ""What is the relationship between items 'A' and 'B' regarding common brands?"". In this case, the user is asking about the paths between two items that is constraint to brands they both belong to.
Suppose the "id"s of the items 'A' and 'B' are 'id1' and 'id2', and the cypher query for this question is (where only paths that contains their belonging brands should be considered):
```cypher
MATCH path = (item1:amazon:item {id: 'id1'})-[:brand]->(brand1:amazon:brand)-[:item]->(item2:amazon:item {id: 'id2'})
RETURN path
LIMIT 10
```

---Goal---

Generate a trustful reasoning path that can be executed on graph using the given user question, based on the given schema of the knowledge graph. Also give the cypher query that can be executed on the graph to get the answer.

If you don't have adequate information to give trustful reasoning paths, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.

---Graph Schema---

{graph_schema}

---Notes---

1. Add the identification label in the generated cypher query as instructed in the graph schema section to ensure the cypher query inquires about the right graph.

2. Use the "id" property to identify the entities in the graph, rather than names. Users will provide the ids of the entities along with the questions.

3. Please carefully think about the query structure and make sure the query is *correct* and *efficient* to execute.
For example, correct cypher query should first "MATCH path" then "RETURN path".
Also, cypher does not allow mixing label expression symbols ('|', '&', '!', and '%') with colon (':') between labels. To match nodes with any type, just use "(:amazon)" is fine.

4. Start from short-path cypher queries and do not use `*1..`, `*1..2`, `*1..3`, `*..` or even larger ranges for elation matching if we don't explicitly tell you to do so as it will take a long time.
If current simple queries can not find the answer, we will ask you to gradually extend the path with specfic path length.

5. Always use "->" rather than "<-" and "-" to indicate the direction of the relationship in the cypher query. Each edge have an reversed edge in the graph, so do not involve duplicated paths.

6. Always use single "MATCH path =" clause for the whole cypher query, starting from one input entity to another and captures the whole path.

7. Return the results in path format:
```cypher
RETURN path
LIMIT 10
```
"""

PROMPTS[
    "direct_cypher_query"
] = """---Role---
You are a helpful assistant that generate cypher queries with the knowledge of the graph schema to find the answer to given questions.

---Setting---

You will be provided with the knowledge graph schema, which indicates the types of nodes and edges, and by what relations nodes are connected.

The user input is a graph question and a id_mapping. The id_mapping is a dictionary that maps the entity names in the question to their corresponding IDs in the database.

---Goal---

Generate the cypher query that can be executed on neo4j database to find the answer to the question based on the provided question and id_mapping.

You should return two columns "source" and "target" using the "id" property, indicating relations consisting of the source and target entity IDs that are required to answer the question: (1) source node entity IDs and (2) target node entity IDs.

If you don't have adequate information to give trustful reasoning paths, just say so. Do not make anything up.

Do not make up any information where the supporting evidence for it is not provided.

---Graph Schema---

{graph_schema}

---Notes---

1. Please carefully think about the query structure and make sure the query is **correct** and **efficient** to execute. Do not forget to assign a variable name before retrieving the attributes.

2. Add the identification label in the generated cypher query as instructed in the graph schema section to ensure the cypher query inquires about the right graph.

3. Use the "id" property to identify the entities in the graph, rather than names. Users will provide the ids of the entities along with the questions.

4. Always use "->" rather than "<-" and "-" to indicate the direction of the relationship in the cypher query. Each edge have an reversed edge in the graph, so do not involve duplicated paths.

5. Only return two columns "souce" and "target" which are the entity IDs of the required relations, using "id" property. Do not involve any other columns or attributes.

6. return the cypher query in the following format:
```cypher
Your cypher query here
```
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["error_retry"] = (
    "When processing you previous response, you have made a mistake and caused an error. Please fix the error and generate the response again. The error is: {}."
)

# PROMPTS["direct_cypher_query_free_form_output"] = """---Role---
# You are a helpful assistant that generate cypher queries with the knowledge of the graph schema to find the answer to given questions.

# ---Setting---

# You will be provided with the knowledge graph schema, which indicates the types of nodes and edges, and by what relations nodes are connected.

# The user input is a graph question and a id_mapping. The id_mapping is a dictionary that maps the entity names in the question to their corresponding IDs in the database.

# ---Goal---

# Generate the cypher query that can be executed on neo4j database to find the answer to the question based on the provided question and id_mapping.

# If you don't have adequate information to give trustful reasoning paths, just say so. Do not make anything up.

# Do not make up any information where the supporting evidence for it is not provided.

# ---Graph Schema---

# Definition of the graph:
# This knowledge graph is a citation graph in physics, there are three types of nodes in this graph: paper, author and venue.

# Node properties:
# 1. type: author, properties: ["id", "name", "node_type"]
# 2. type: paper, properties: ["id", "name", "label", "year", "node_type", "abstract"]
# 3. type: venue, properties: ["id", "name", "node_type"]

# Edge properties:
# Author nodes are linked to their paper nodes by authorship. Specific relations are:
# 1. author -> "paper" -> paper

# Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Specific relations are:
# 1. paper -> "author" -> author
# 2. paper -> "reference" -> paper
# 3. paper -> "venue" -> venue
# 3. paper -> "cited_by" -> paper

# Venue nodes are linked to their included paper nodes. Specific relations are:
# 1. venue -> "paper" -> paper

# ---Notes---

# 1. Please carefully think about the query structure and make sure the query is **correct** and **efficient** to execute. Do not forget to assign a variable name before retrieving the attributes.

# 2. Add the identification label "Physics" to the entities, for example, ":physics:author" (same for other types of entities).

# 3. Use the "id" property to identify the entities in the graph, rather than names. Users will provide the ids of the entities along with the questions.

# 4. Always use "->" rather than "<-" and "-" to indicate the direction of the relationship in the cypher query. Each edge have an reversed edge in the graph, so do not involve duplicated paths.

# 5. Return type could be either entity IDs or paths of entities and relations, depending on what the question is asking for.
# """
