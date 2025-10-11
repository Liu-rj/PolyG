"""
Reference:
 - Some Prompts are based on [graphrag](https://github.com/microsoft/graphrag) and [nano-graphrag](https://github.com/gusye1234/nano-graphrag)
"""

PHYSICS_GRAPH_SCHEMA = """
Definition of the graph:
This knowledge graph is a citation graph in physics, there are three types of nodes in this graph: paper, author and venue.

Node properties:
1. type: author, properties: ["id", "name", "node_type"]
2. type: paper, properties: ["id", "name", "label", "year", "node_type", "abstract"]
3. type: venue, properties: ["id", "name", "node_type"]

Edge properties (all relation names are single-directional):
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

Edge properties (all relation names are single-directional):
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

Edge properties (all relation names are single-directional):
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

FREEBASE_GRAPH_SCHEMA = """
Definition of the graph:
This knowledge graph is a subgraph extracted from freebase, there are many types of relations connecting the nodes.

Node properties:
1. type: node, properties: ["id", "name"]

Edge properties:
In freebase, nodes are linked to each other through various relations and domains. Specific relation types are:
{schema}

Example relations:
{example}

Graph indentifier:
Add the identification label "{benchmark}" to the entities in the cypher query, for example, ":{benchmark}:node" for the entities in the graph.
"""

SCHEMA_MAP = {
    "physics": PHYSICS_GRAPH_SCHEMA,
    "amazon": AMAZON_GRAPH_SCHEMA,
    "goodreads": GOODREADS_GRAPH_SCHEMA,
    "webqsp": FREEBASE_GRAPH_SCHEMA,
    "cwq": FREEBASE_GRAPH_SCHEMA,
}

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS[
    "local_rag_response"
] = """---Role---

You are a helpful assistant responding to user questions.

---Setting---

You will be provided with some helpful references. The provided reference data tables may appear in the following forms:
1. The cypher query used to retrieve the reference data.
1. Three tables are provided: 1. entity table, 2. relation table, and 3. reasoning paths indicating relations between entities.

Questions can be about node inquiries or relations between nodes. Entities in the question may not have direct relationship and answering the question will need to consider multi-hop relations.

---Goal---

Generate a response to the user's question following the given target length and format.

Note:
1. Users may not see the provided data tables, so provide necessary evidence in your response.
2. If you don't know the answer, just say so. Do not make anything up.

---Target response length and format---

{response_type}

---Data tables---

{context_data}
"""

PROMPTS[
    "guided_walk_response"
] = """---Role---

You are a helpful assistant responding to user questions.

---Setting---

You will be provided with some helpful references. The provided reference data tables may appear in the following forms:
1. Three tables are provided: 1. entity table, 2. relation table, and 3. reasoning paths.
2. Multiple reasoning paths are provided, indicating relations between entities.

---Goal---

Generate a response to the user's question following the given target length and format.

If the reference tables do not provide sufficient evidence, use your internal knowledge to answer the question.

---Target response length and format---

{response_type}

---Data tables---

{context_data}
"""

PROMPTS[
    "cypher_answer_summary"
] = """---Role---

You are a helpful assistant summarize a comprehensive answer to questions using the answer tables provided.

---Setting---

You will be provided with an answer table, which contains the answer entities to the user question as each row in the answer table. The concrete relations between the entities and the question entities are not given in the answer table, you don't need to infer or disclaim them.

You should give a comprehensive summary based on the answers in naturally languages with rich founding knowledge and informative contents based on the attributes of the rows in the data table.

For example, if the question is "Who are the the authors of the book 'xxx'?", then the rows in the answer table are the author entities as the answers and the book entities. Based on the answer table, you should give a summary to introduce who are the authors with contents in the answer table.

You don't need to come up with the answers yourself as they are already given, just give a summary based on the answers.

Note: Keep the name of answer entities as they are in the answer table, do not change them in any way.

---Goal---

The summary should be comprehensive, diverse and empowerful, which can thoroughly cover diverse aspects. Provide as much detail as you can from the data table, and enable the reader to understand the topic and make informed judgments.

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

Given user question, generate a cypher query that can be executed on the neo4j database to find the reasoning paths based on the given schema of the knowledge graph.

If you don't have adequate information to give trustful reasoning paths, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.

---Example---

1. User query: What are the titles of all papers authored by 'Hee-Dong Jeong'?
Suppose the id of 'Hee-Dong Jeong' is '2901004538', the cypher query for this question is:
```cypher
MATCH (a:physics:author {{id: '2901004538'}})-[:paper]->(p:physics:paper)
RETURN DISTINCT p.id AS id
LIMIT 20
```

---Graph Schema---

{graph_schema}

---Notes---

1. Please carefully think about the query structure and make sure the query is **correct** and **efficient** to execute. Do not forget to assign a variable name before retrieving the attributes.

2. Add the identification label in the generated cypher query as instructed in the graph schema section to ensure the cypher query inquires about the right graph.

3. Use the "id" property to identify the entities in the graph, rather than names. Users will provide the ids of the entities along with the questions.

4. Always use "->" rather than "<-" and "-" to indicate the direction of the relationship in the cypher query. Each edge have an reversed edge in the graph, so do not involve duplicated paths.

5. Only return the unique id column of retrieved entities and set the label of the result column as "id", using `RETURN DISTINCT node.id as id` in the cypher query. Do not return other properties or columns, i.e., `RETURN DISTINCT node.name as id` is not allowed.

6. Use "LIMIT 20" to limit the number of results returned in the cypher query.

7. return the cypher query in the following format:
```cypher
Your cypher query here
```
"""


PROMPTS[
    "cypher_path_search_prompt"
] = """---Role---

You are a helpful assistant that can generate trustful reasoning paths with the knowledge of the graph schema to answer the given user question.

---Setting---

You will be provided with the knowledge graph schema, which indicates the types of nodes and edges, and by what relations nodes are connected.

User questions are about relationships between nodes which can be indirect and result in multi-hop relation paths. User questions may include a relation constraint that indicates some specific paths.

---Goal---

Generate a trustful reasoning path that can be executed on graph using the given user question, based on the given schema of the knowledge graph. Also give the cypher query that can be executed on the graph to get the answer.

If you don't have adequate information to give trustful reasoning paths, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.

---Examples---

1. User question: "What is the relationship between 'J. Koll' and 'Z. Staykova' in terms of paper reference?". In this case, the user is asking about the paths between two authors that is constraint to paper references and citations.
Suppose the "id" of 'J. Koll' and 'Z. Staykova' is 'id1' and 'id2', and the cypher query for this question is (where only paths that contains reference and citation relations should be considered):
```cypher
MATCH path = (author1:author {{id: 'id1'}})-[:paper]->(paper1:paper)-[:reference|cited_by]->(paper2:paper)-[:author]->(author2:author {{id: 'id2'}})
RETURN path
LIMIT 20
```

2. User question: "What is the relationship between authors 'A' and 'B' regarding collaborated books?". In this case, the user is asking about the paths between two authors that is constraint to books they have co-authored.
Suppose the "id"s of author 'A' and 'B' are 'id1' and 'id2', and the cypher query for this question is (where only paths that contains co-authored books should be considered):
```cypher
MATCH path = (author1:author {{id: 'id1'}})-[:book]->(book1:book)-[:author]->(author2:author {{id: 'id2'}})
RETURN path
LIMIT 20
```

3. User question: ""What is the relationship between items 'A' and 'B' regarding common brands?"". In this case, the user is asking about the paths between two items that is constraint to brands they both belong to.
Suppose the "id"s of the items 'A' and 'B' are 'id1' and 'id2', and the cypher query for this question is (where only paths that contains their belonging brands should be considered):
```cypher
MATCH path = (item1:item {{id: 'id1'}})-[:brand]->(brand1:brand)-[:item]->(item2:item {{id: 'id2'}})
RETURN path
LIMIT 20
```

---Graph Schema---

{graph_schema}

---Notes---

1. Add the identification label in the generated cypher query as instructed in the graph schema section to ensure the cypher query inquires about the right graph.

2. Use the "id" property to identify the entities in the graph, rather than names. Users will provide the ids of the entities along with the questions.

3. Please carefully think about the query structure and make sure the query is *correct* and *efficient* to execute. For example, correct cypher query should first "MATCH path" then "RETURN path". Also, cypher does not allow mixing label expression symbols ('|', '&', '!', and '%') with colon (':') between labels. Make sure to use the relation with its correct direction (-> or <-) in the cypher query.

4. Start from short-path cypher queries and do not use `*1..`, `*1..2`, `*1..3`, `*..` or even larger ranges for elation matching if we don't explicitly tell you to do so as it will take a long time. For example, just starting from "(:paper)-[:reference|cited_by]->(:paper)" for matching "paper reference" relations is good. If current simple queries can not find the answer, we will ask you to gradually extend the path with specfic path length.

5. IMPORTANT: Always use single "MATCH path =" clause for the whole cypher query, starting from one input entity to another and captures the whole path, and return the "path". Results are extracted using the "path" variable, so program will fail if you return other variables. Do not split the path matching into multiple "MATCH" clauses, just use a single path matching and return it!

6. Return the results in path format (enclosed by ```cypher ... ```):
```cypher
RETURN path
LIMIT 20
```
"""


PROMPTS[
    "cypher_only_query"
] = """---Role---
You are a helpful assistant that generate cypher queries with the knowledge of the graph schema to find the answer to given questions.

---Setting---

You will be provided with the knowledge graph schema, which indicates the types of nodes and edges, and by what relations nodes are connected.

The user input is a graph question and a id_mapping. The id_mapping is a dictionary that maps the entity names in the question to their corresponding IDs in the database.

---Goal---

Generate the cypher query that can be executed on neo4j database to find the answer to the question based on the provided question and id_mapping.

If you don't have adequate information to give trustful reasoning paths, just say so. Do not make anything up.

Do not make up any information where the supporting evidence for it is not provided.

---Graph Schema---

{graph_schema}

---Notes---

1. Please carefully think about the query structure and make sure the query is **correct** and **efficient** to execute. Do not forget to assign a variable name before retrieving the attributes. Make sure to use the relation with its correct direction (-> or <-) in the cypher query.

2. Add the identification label in the generated cypher query as instructed in the graph schema section to ensure the cypher query inquires about the right graph.

3. Use the "id" property to identify the entities in the graph, rather than names. Users will provide the ids of the entities along with the questions.

4. return the cypher query in the following format (enclosed by ```cypher ... ```):
```cypher
Your cypher query here
```
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["token_limit_exceeded"] = (
    "Content exceeds the token limit, please reduce the content size and try again."
)

PROMPTS["error_retry"] = (
    "When processing your previous response, errors occurred which indicates that you have made a mistake. Please fix the error and generate the response again. The error is: {}."
)

PROMPTS[
    "question_classification"
] = """
**Prompt:**
You are an intelligent assistant tasked with classifying questions into different types based on the missing parts of a fact and the nature of the aspect being asked.

We have five types of questions, where s is the subject, p is the predicate, o is the object, and * means the missing part. Below, we define the five types of questions with some concrete examples for each type.

The five types of questions are:
- **<s,*,*> (0):**
  In this type, the object and concrete predicate are both missing in the facts and question asks about the general or broad information of the subject (e.g., themes, concepts, or a general description).
  Examples:
  - "Tell me about 'The Woman in Black: A Ghost Play'."
  - "Give a broad description about 'Frankenstein, or The Modern Prometheus'."
  - "Who is 'Barack Obama'?"
  - "Who is flo from progressive?"
- **<s,p,*> (1):**
  In this type, the object is missing in the facts and question focuses on finding the object, with specific and concrete relations and attributes to the subject are provided by a concrete predicate p.
  Note that this type of question can have multiple predicates (a predicate chain) and subjects, and containing time or counting constraints in the predicates.
  Examples:
  - "Who are the authors of the book 'Sunshine for the Latter-Day Sa'?"
  - "What series have the author of the book 'Cookies for the Dragon (Saint Lakes, #2.1)' published?"
  - "Who are the authors of the books that are published by the publishers that have published books of the series 'Shifter Justice'?"
  - "What are the 5 biggest cities in the usa?"
  - "During what war did abraham lincoln serve as president?"
  - "which city held the summer olympics twice?"
  - Example for multiple subjects and predicates: "What are the 5 biggest cities in the usa and have a population of more than 1 million?"
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
  - "Have the authors 'Rubem Fonseca' and 'Lygia Fagundes Telles' ever published books in the same publishers?"
  - "Do the publishers 'Scholastic Inc.' and 'Klutz' have any authors publishing books in both of them?"
- **nested (-1):**
  Not every question can be directly classified into the above four classes, especially for complex questions, where the question is nested and asks about different types of relations, in which case you should return **-1**.
  For example, when the question asks about the relationship between two entities or inquiry about general information about some entities, but the entities need to be determined by another query embeded in the overall question, it is a nested question and should be classified as -1.
  Especially, nested questions can only be in the form where the overall quesiton is one of <s,*,*>, <s,*,o>, <s,p,o> question nested with <s,p,*> question.
  Examples:
  - "Provide some information about the academic work and achievements of the scholars who have collaborated with 'L. Foldy'.": 1. the first step is a <s,p,*> question, finding the academic collaborators of 'L. Foldy'. 2. the second step is a <s,*,*> question: gather general academic information for the collaborators.
  - "Have the scholars 'S. Chiku' and 'A. I. Sanda' both published work at the venue having the paper 'weyl groups in ads3 cft2'?": 1. the first step is a <s,p,o> question, finding the venue having the paper 'weyl groups in ads3 cft2'. 2. the second step is a <s,p,o> question, finding the path validating whether the authors both published work in the venue found in the previous step.
  - "In what way is the scholar 'Liang Wu' linked to the writers of 'bound states of breathing airy gaussian beams in nonlocal nonlinear medium'?" 1. the first step is a <s,p,*> question: find the authors of the paper 'bound states of breathing airy gaussian beams in nonlocal nonlinear medium'. 2. then a <s,*,o> question: find the path validating whether 'Liang Wu' is linked to the authors found in the previous step.

**Differences between <s,p,*> and <s,p,o>:**
In the case of multiple predicates and subjects, <s,p,*> questions may sound similar to <s,p,o> questions, but they are different in what they inquiry about.
If the question asks for some concrete entities, for example "What is/are the entities that ...?", then that is <s,p,*> question.
If the questions focus on checking the existence of a relationship between two entities, for example "Do A and B share ...?", then that is <s,p,o> question.

**Instruction:**
When given a question, analyze its type based on the definitions above, then return **only a single number** corresponding to the type of the question.

Notes:
1. If the question reveals a chain of specific relations (multi-hop predicates) from one specific entity (subject), it is not a nested question and should be classified as <s,p,*> question and output 1.
2. In the case of multiple sequential <s,p,*> sub-questions with different subjects, you can merge them into one single <s,p,*> question and classify the overall question as 1 for <s,p,*>, as consecutive <s,p,*> questions can always be merged into one single <s,p,*> question with multiple predicates and subjects.


For the response, you don't need to give any explanation but just **a single number** indicating the type:

- **0** for <s,*,*>
- **1** for <s,p,*>
- **2** for <s,*,o>
- **3** for <s,p,o>
- **-1** for nested questions.

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
  In this type, the object and concrete predicate are both missing in the facts and question asks about the general or broad information of the subject (e.g., themes, concepts, or a general description).
  Examples:
  - "Tell me about 'The Woman in Black: A Ghost Play'."
  - "Give a broad description about 'Frankenstein, or The Modern Prometheus'."
  - "Who is 'Barack Obama'?"
  - "Who is flo from progressive?"
- **<s,p,*> (1):**
  In this type, the object is missing in the facts and question focuses on finding the object, with specific and concrete relations and attributes to the subject are provided by a concrete predicate p.
  Note that this type of question can have multiple predicates and subjects, and containing time or counting constraints in the predicates.
  Examples:
  - "Who are the authors of the book 'Sunshine for the Latter-Day Sa'?"
  - "What series have the author of the book 'Cookies for the Dragon (Saint Lakes, #2.1)' published?"
  - "Who are the authors of the books that are published by the publishers that have published books of the series 'Shifter Justice'?"
  - "What are the 5 biggest cities in the usa?"
  - "During what war did abraham lincoln serve as president?"
  - "which city held the summer olympics twice?"
  - Example for multiple subjects and predicates: "What are the 5 biggest cities in the usa and have a population of more than 1 million?"
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
  - "Have the authors 'Rubem Fonseca' and 'Lygia Fagundes Telles' ever published books in the same publishers?"
  - "Do the publishers 'Scholastic Inc.' and 'Klutz' have any authors publishing books in both of them?"

**Differences between <s,p,*> and <s,p,o>:**
In the case of multiple predicates and subjects, <s,p,*> questions may sound similar to <s,p,o> questions, but they are different in what they inquiry about.
If the question asks for some concrete entities, for example "What is/are the entities that ...?", then that is <s,p,*> question.
If the questions focus on checking the existence of a relationship between two entities, for example "Do A and B share ...?", then that is <s,p,o> question.

**Instruction:**
When given a nested question that can not be directly classified into one of the four types, decompose it into a plan of several sub-questions with each being a unit question based on the definitions above.
We will also provide you the graph schema, which indicates the types of nodes and edges, and by what relations nodes are connected. This will help you to identify the sub-questions and generate a description for each step of the decomposition plan.
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
1. <s,p,*>: find the venue having the paper 'weyl groups in ads3 cft2'.
2. <s,p,o>: find the path validating whether the authors both published work in the venue found in the previous step.

Important Note: Always avoid consecutive <s,p,*> steps, they can be merged into one single <s,p,*> question with a chain of relations (multi-hop), and you should merge them into one step, just like the examples shown in demonstration of the <s,p,*> question ("Who are the authors of the books that are published by the publishers that have published books of the series 'Shifter Justice'?").

**Graph Schema:**
{graph_schema}

**Question:** {query}

Return your answer in the following format:
```plan
1. question-type: desciption 1
2. question-type: desciption 2
...
```
"""

PROMPTS[
    "nested_query_instantiation"
] = """
**Prompt:**
You are an intelligent assistant tasked with instantiating concrete questions for a step of a nested question based on its given question decomposition plan and responses to previous steps' questions.

You should generate a list of conrete questions based on the current step of the plan and also a corresponding entity id-name mapping for the entities in each question.

**Instruction:**
The input will be five parts:
1. The nested question, which is the original overall question that needs to be answered.
2. The question plan, which is a list of steps with each step being a description for a unit question.
3. The step we are in, indicating which step of the plan we should generate this concrete question for.
4. The previous step's response, which is the answer to the previous step's question and should be used for instantiating current concrete question. This may be empty if there is not previous step.
5. The entity and name mapping, which is a dictionary that maps the entity names in the question to their corresponding IDs in the database.

**Note:**
- In the following question type, s means subject, p means predicate, o means object, and * means the missing part.
- For <s,*,*> and <s,p,*> question, you should merge several questions into one question. For example two <s,*,*> questions, "Tell me about 'A'." and "Tell me about 'B'." can be merged into "Tell me about 'A' and 'B'.". And you should give both the id mapping for 'A' and 'B'. Same for <s,p,*> questions, for example "Who is the collaborator of both 'A' and 'B'?".
- For other two types of questions <s,*,o> and <s,p,o>, you need to generate a seperate concrete question for each entity. For example, "What is the relation between 'L. Foldy' and 'A'?" and "What is the relation between 'L. Foldy' and 'B'?" should be generated as two separate questions. And you should give a seperate id mapping for each concrete question.

**Example:**

Example 1:
Nested Question: "Tell me about the academic contributions of the academic collaborators of the scholar 'L. Foldy'."

Question plan:
1. <s,p,*>: Find all academic collaborators who have worked with 'L. Foldy'.
2. <s,*,*>: For each collaborator identified in step 1, retrieve their academic contributions and achievements.

Step: 2

Previous step's response:
1. academic collaborators who have worked with 'L. Foldy' are "A", "B", and "C".

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
1. authors of the paper 'kinetic and mass mixing with three abelian groups' are "A" and "B".

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

-----
Think carefully before you answer, remember to merge questions for <s,*,*> and <s,p,*> question types, and generate seperate questions for <s,*,o> and <s,p,o> question types.
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
