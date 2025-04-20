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
MATCH path = (author1:author {{id: 'id1'}})-[:paper]->(paper1:paper)-[:reference|cited_by]->(paper2:paper)-[:author]->(author2:author {{id: 'id2'}})
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
MATCH path = (author1:author {{id: 'id1'}})-[:book]->(book1:book)-[:author]->(author2:author {{id: 'id2'}})
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

4. Start from short-path cypher queries and do not use `*1..`, `*1..2`, `*1..3`, `*..` or even larger ranges for relation matching if we don't explicitly tell you to do so as it will take a long time.
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
MATCH path = (item1:amazon:item {{id: 'id1'}})-[:brand]->(brand1:amazon:brand)-[:item]->(item2:amazon:item {{id: 'id2'}})
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
    "When processing your previous response, errors occurred which indicates that you have made a mistake. Please fix the error and generate the response again. The error is: {}."
)

PROMPTS[
    "question_classification"
] = """
**Prompt:**
You are an intelligent assistant tasked with classifying questions into different types based on the missing parts of a fact and the nature of the aspect being asked.

We have four types of questions, where s is the subject, p is the predicate, o is the object, and * means the missing part. Below, we define the four types of questions with some concrete examples for each type.

The four types of questions are:
- **<s,*,*> (0):**
  In this type, the object and concrete predicate are both missing in the facts and question asks about the general or abstract information of the subject (e.g., themes, concepts, or a general description).
  Examples:
  - "Tell me about 'The Woman in Black: A Ghost Play'."
  - "Tell me about 'Frankenstein, or The Modern Prometheus'."
  - "Who is 'Barack Obama'?"
  - "Who is flo from progressive?"
- **<s,p,*> (1):**
  In this type, the object is missing in the facts and question focuses on finding the object, which is some specific and concrete details related to the subject (e.g., relations or other direct attributes) with a concrete predicate p. There can be multiple predicates and subjects in the question, containing time or counting constraints.
  Examples:
  - "Who are the authors of the book 'Sunshine for the Latter-Day Sa'?"
  - "What series have the author of the book 'Cookies for the Dragon (Saint Lakes, #2.1)' published?"
  - "What are the authors of the books that are published by the publishers that have published books of the series 'Shifter Justice'?"
  - "What are the 5 biggest cities in the usa?"
  - "During what war did abraham lincoln serve as president?"
  - "which city held the summer olympics twice?"
- **<s,*,o> (2):**
  In this type, both the subject and object are given but the predicate is missing. Question asks about their relationships or interactions (e.g., conceptual connections, influences).
  Examples:
  - "What is the relationship between 'Kizuna' and 'Kazuma Kodaka'?"
  - "What is the relationship between 'Monster House Press' and 'Matt Hart'?"
  - "What are the impacts of information communication technology to public relation practices?"
  - "How are the directions of the velocity and force vectors related in a circular motion?"
  - "HOW AFRICAN AMERICANS WERE IMMIGRATED TO THE US?"
- **<s,p,o> (3):**
  In this question, the subject, predicates object are all given. Questions either inquires about particular aspects of the relationship between the entities or verifies whether a specific relationship exists.
  Examples:
  - "Have the authors 'Rubem Fonseca' and 'Lygia Fagundes Telles' ever published books in the same publishers? If so, tell me some examples."
  - "Do the publishers 'Scholastic Inc.' and 'Klutz' have any authors publishing books in both of them and what are the publications and authors?"

**Instruction:**
When given a question, analyze it based on the definitions above. If the question belongs to any of the four types, then return **only a single number** corresponding to the type of the question.
However, note that not every question can be directly classified into the above four classes, where the question is nested and asks about different types of relations, in which case you should return **-1**.
For example, when the question asks about the relationship between two relations but one entity needs to be determined by another query embeded in the overall question, it is a nested question and should be classified as -1.

Some concrete example:
- "Tell me about the academic collaborators of the scholar 'L. Foldy'.": 1. the first step is a <s,p,*> question, finding the collaborators of 'L. Foldy'. 2. the second step is a <s,*,*> question: gather general information for the collaborators.
- "Have the scholars 'S. Chiku' and 'A. I. Sanda' both published work at the venue having the paper 'weyl groups in ads3 cft2'?": 1. the first step is a <s,p,o> question, finding the venue having the paper 'weyl groups in ads3 cft2'. 2. the second step is a <s,p,o> question, finding the path validating whether the authors both published work in the venue found in the previous step.
- "In what way is the scholar 'Liang Wu' linked to the writers of 'bound states of breathing airy gaussian beams in nonlocal nonlinear medium'?" 1. the first step is a <s,p,*> question: find the authors of the paper 'bound states of breathing airy gaussian beams in nonlocal nonlinear medium'. 2. then a <s,*,o> question: find the path validating whether 'Liang Wu' is linked to the authors found in the previous step.


For the response, you don't need to give any explanation but just **a single number** indicating the type:

- **0** for <s,*,*>
- **1** for <s,p,*>
- **2** for <s,*,o>
- **3** for <s,p,o>
- **-1** for questions that do not belong to any of the four types.

**Question:** {}

**Your Answer:**
[a single number here]
"""

PROMPTS[
    "nested_query_decomposition"
] = """
**Prompt:**
You are an intelligent assistant tasked with decomposing nested questions into a plan of several sub-questions using the following question taxonomy.

We have four types of unit questions, where s is the subject, p is the predicate, o is the object, and * means the missing part. Below, we define the four types of questions with some concrete examples for each type.

The four types of questions are:
- **<s,*,*> (0):**
  In this type, the object and concrete predicate are both missing in the facts and question asks about the general or abstract information of the subject (e.g., themes, concepts, or a general description).
  Examples:
  - "Tell me about 'The Woman in Black: A Ghost Play'."
  - "Tell me about 'Frankenstein, or The Modern Prometheus'."
  - "Who is 'Barack Obama'?"
  - "Who is flo from progressive?"
- **<s,p,*> (1):**
  In this type, the object is missing in the facts and question focuses on finding the object, which is some specific and concrete details related to the subject (e.g., relations or other direct attributes) with a concrete predicate p. There can be multiple predicates and subjects in the question, containing time or counting constraints.
  Examples:
  - "Who are the authors of the book 'Sunshine for the Latter-Day Sa'?"
  - "What series have the author of the book 'Cookies for the Dragon (Saint Lakes, #2.1)' published?"
  - "What are the authors of the books that are published by the publishers that have published books of the series 'Shifter Justice'?"
  - "What are the 5 biggest cities in the usa?"
  - "During what war did abraham lincoln serve as president?"
  - "which city held the summer olympics twice?"
- **<s,*,o> (2):**
  In this type, both the subject and object are given but the predicate is missing. Question asks about their relationships or interactions (e.g., conceptual connections, influences).
  Examples:
  - "What is the relationship between 'Kizuna' and 'Kazuma Kodaka'?"
  - "What is the relationship between 'Monster House Press' and 'Matt Hart'?"
  - "What are the impacts of information communication technology to public relation practices?"
  - "How are the directions of the velocity and force vectors related in a circular motion?"
  - "HOW AFRICAN AMERICANS WERE IMMIGRATED TO THE US?"
- **<s,p,o> (3):**
  In this question, the subject, predicates object are all given. Questions either inquires about particular aspects of the relationship between the entities or verifies whether a specific relationship exists.
  Examples:
  - "Have the authors 'Rubem Fonseca' and 'Lygia Fagundes Telles' ever published books in the same publishers? If so, tell me some examples."
  - "Do the publishers 'Scholastic Inc.' and 'Klutz' have any authors publishing books in both of them and what are the publications and authors?"

**Instruction:**
When given a nested question that can not be directly classified into one of the four types, decompose it into a plan of several sub-questions with each being a unit question based on the definitions above.
For each step of the decomposition plan, you should first identify the type of the sub-question and then generate a description which will further guide the instantiation of the concrete sub-question.

For example:
Question: "Tell me about the academic contributions of the academic collaborators of the scholar 'L. Foldy'."
Decomposition:
1. <s,p,*>: find the academic collaborators of the scholar 'L. Foldy'.
2. <s,*,*>: retrieve the academic contributions of the scholars found in the previous step.

Question: "In what way is the scholar 'Liang Wu' linked to the writers of 'bound states of breathing airy gaussian beams in nonlocal nonlinear medium'?"
Decomposition:
1. <s,p,*>: find the authors of the paper 'bound states of breathing airy gaussian beams in nonlocal nonlinear medium'.
2. <s,*,o>: find the path validating whether 'Liang Wu' is linked to the authors found in the previous step.

Question: "Have the scholars 'S. Chiku' and 'A. I. Sanda' both published work at the venue having the paper 'weyl groups in ads3 cft2'?"
Decomposition:
1. <s,p,o>: find the venue having the paper 'weyl groups in ads3 cft2'.
2. <s,p,o>: find the path validating whether the authors both published work in the venue found in the previous step.

**Question:** {}

Return your answer in the following format:
```plan
1. question-type: desciption 1
2. question-type: desciption 2
```
...
"""

PROMPTS[
    "nested_query_instantiation"
] = """
**Prompt:**
You are an intelligent assistant tasked with instantiating concrete questions for a step of a nested question based on its given question decomposition plan and responses to previous steps' questions.

You should generate a list of conrete questions based on the current step of the plan and also a corresponding entity id-name mapping for the entities in each question.

**Instruction:**
The input will be four parts:
1. The nested question, which is the original overall question that needs to be answered.
2. The question plan, which is a list of steps with each step being a description for a unit question.
3. The step we are in, indicating which step of the plan we should generate this concrete question for.
4. The previous step's response, which is the answer to the previous step's question and should be used for instantiating current concrete question. This may be empty if there is not previous step.
5. The entity and name mapping, which is a dictionary that maps the entity names in the question to their corresponding IDs in the database.

**Note:**
- In the following question type, s means subject, p means predicate, o means object, and * means the missing part.
- For <s,*,*> question, you can merge several questions into one question. For example, "Tell me about 'A'." and "Tell me about 'B'." can be merged into "Tell me about 'A' and 'B'.". And you should give both the id mapping for 'A' and 'B'.
- For other types of questions, you need to generate a seperate concrete question for each entity. For example, "What is the relation between 'L. Foldy' and 'A'?" and "What is the relation between 'L. Foldy' and 'B'?" should be generated as two separate questions. And you should give a seperate id mapping for each concrete question.

**Example:**

Example 1:
Nested Question: "Tell me about the academic contributions of the academic collaborators of the scholar 'L. Foldy'."

Question plan:
1. <s,p,*>: Find all academic collaborators who have worked with 'L. Foldy'.
2. <s,*,*>: For each collaborator identified in step 1, retrieve their academic contributions and achievements.

Step: 2

Previous step's response:
1. ["A", "B", "C"]

Id mapping:
{{"A": "id1", "B": "id2", "C": "id3"}}

Generated concrete question for current step:
```json
{{
    "question1": {{
        "question": "Tell me about the academic contributions and achievements of the scholars 'A', 'B', and 'C'.",
        "id_mapping": {{ 'A': 'id1', 'B': 'id2', 'C': 'id3' }}
    }},
}}
```
""

Example 2:
Nested Question: "What is the relationship between the scholar 'Steven D. Bass' and the authors of the paper 'kinetic and mass mixing with three abelian groups'?"

Question plan:
1. <s,p,*>: Find the authors of the paper 'kinetic and mass mixing with three abelian groups'.
2. <s,*,o>: Find the relationship between 'Steven D. Bass' and the authors identified in step 1.

Step: 2

Previous step's response:
1. ["A", "B"]

Id mapping:
{{"A": "id1", "B": "id2"}}

Generated concrete question for current step:
```json
{{
    "question1": {{
        "question": "What is the relationship between the scholar 'Steven D. Bass' and the author 'A'?",
        "id_mapping": {{ "Steven D. Bass": "id_steven", "A": "id1" }}
    }},
    "question2": {{
        "question": "What is the relationship between the scholar 'Steven D. Bass' and the author 'B'?",
        "id_mapping": {{ "Steven D. Bass": "id_steven", "B": "id2" }}
    }},
}}
```


**Task:**
Nested question: {question}

Question plan: {query_plan}

Step: {step}

Previous step's response:
{history}

Entity ID mapping:
{mapping}

Return your answer in the following format:
```json
{{
    "question1": {{
        "question": "[your question here]",
        "id_mapping": [your id mapping here]
    }},
    ...
}}
```
"""


PROMPTS[
    "nested_query_summarization"
] = """
**Prompt:**
You are an intelligent assistant tasked with summarizing a final response for the given nested question based on decomposed sub-questions and the responses to them.

**Instruction:**
The input will be three parts:
1. The nested question, which is the original overall question that needs to be answered.
2. The question plan, which is a list of steps with each step being a description for a sub-question.
3. The responses to each step, which are the answers to the sub-questions and should be used for summarizing the final response.

**Task:**
Nested question: {question}

Question plan: {query_plan}

Responses to each step:
{history}
"""


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
