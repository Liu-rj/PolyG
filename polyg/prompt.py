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

WEBQSP_GRAPH_SCHEMA = """
Definition of the graph:
This knowledge graph is a subgraph extracted from freebase, there are many types of relations connecting the nodes.

Node properties:
1. type: node, properties: ["id", "name"]

Edge properties:
In freebase, nodes are linked to each other through various relations and domains. Specific relation types are:
1. book_written_work_subjects
2. government_national_anthem_national_anthem_of
3. common_webpage_resource
4. location_statistical_region_gender_balance_members_of_parliament
5. olympics_olympic_medal_honor_medal
6. sports_competitor_country_relationship_sports
7. location_partial_containment_relationship_partially_contained_by
8. organization_organization_scope_organizations_with_this_scope
9. olympics_olympic_athlete_affiliation_sport
10. location_statistical_region_gross_savings_as_percent_of_gdp
11. location_administrative_division_capital
12. location_statistical_region_diesel_price_liter
13. olympics_olympic_medal_honor_medalist
14. common_topic_notable_for
15. government_government_office_or_title_office_holders
16. symbols_flag_used_by
17. organization_organization_founder_organizations_founded
18. location_statistical_region_gni_per_capita_in_ppp_dollars
19. location_location_partially_contains
20. base_aareas_schema_administrative_area_subdividing_type
21. people_person_nationality
22. film_film_location_featured_in_films
23. government_government_position_held_office_position_or_title
24. base_popstra_celebrity_vacations_in
25. location_statistical_region_poverty_rate_2dollars_per_day
26. type_type_expected_by
27. government_government_office_or_title_category
28. government_national_anthem_of_a_country_anthem
29. location_country_currency_used
30. time_event_included_in_event
31. sports_competitor_country_relationship_competitor
32. organization_organization_membership_member
33. business_employment_tenure_company
34. meteorology_cyclone_affected_area_cyclones
35. government_governmental_jurisdiction_governing_officials
36. sports_competitor_competition_relationship_competition
37. sports_sports_team_sport
38. symbols_name_source_namesakes
39. sports_sport_country_athletes
40. location_country_administrative_divisions
41. location_statistical_region_net_migration
42. olympics_olympic_sport_athletes
43. award_award_presented_by
44. meteorology_tropical_cyclone_tropical_cyclone_season
45. people_person_employment_history
46. measurement_unit_dated_percentage_source
47. type_property_schema
48. base_ontologies_ontology_instance_mapping_freebase_topic
49. location_location_nearby_airports
50. symbols_flag_use_flag_user
51. location_location_geolocation
52. olympics_olympic_medal_honor_country
53. sports_tournament_event_competitor_events_competed_in
54. base_locations_countries_continent
55. location_location_containedby
56. location_statistical_region_foreign_direct_investment_net_inflows
57. government_government_position_held_office_holder
58. sports_tournament_event_competition_competitors
59. government_form_of_government_countries
60. government_politician_government_positions_held
61. sports_sport_country_athletic_performances
62. location_country_official_language
63. location_statistical_region_deposit_interest_rate
64. common_topic_webpage
65. sports_multi_event_tournament_sports
66. location_administrative_division_country
67. base_popstra_vacation_choice_vacationer
68. base_aareas_schema_administrative_area_type_subdivides_place
69. government_government_position_held_basic_title
70. base_ontologies_ontology_instance_mapping_ontology
71. freebase_valuenotation_is_reviewed
72. sports_tournament_team_tournaments_competed_in
73. location_country_national_anthem
74. location_statistical_region_part_time_employment_percent
75. business_employment_tenure_person
76. olympics_olympic_games_sports
77. base_aareas_administrative_area_level_examples
78. location_partial_containment_relationship_partially_contains
79. base_culturalevent_event_entity_involved
80. location_statistical_region_cpi_inflation_rate
81. olympics_olympic_sport_olympic_games_contested
82. organization_membership_organization_members
83. sports_competitor_competition_relationship_tournament
84. government_governmental_body_jurisdiction
85. sports_competitor_competition_relationship_country
86. meteorology_tropical_cyclone_affected_areas
87. location_statistical_region_gdp_nominal_per_capita
88. location_statistical_region_labor_participation_rate
89. olympics_olympic_medal_honor_event
90. fictional_universe_fictional_setting_fictional_characters_born_here
91. people_person_languages
92. award_award_presenting_organization_awards_presented
93. government_government_agency_jurisdiction
94. sports_sports_team_location
95. government_government_position_held_jurisdiction_of_office
96. food_beer_from_region
97. common_webpage_in_index
98. book_written_work_original_language
99. location_administrative_division_capital_relationship_capital
100. location_statistical_region_agriculture_as_percent_of_gdp
101. base_aareas_schema_administrative_area_administrative_parent
102. location_statistical_region_prevalence_of_undernourisment
103. location_statistical_region_size_of_armed_forces
104. common_topic_notable_types
105. base_athletics_athletics_championships_competition_athlete_relationship_event
106. book_written_work_author
107. geography_river_basin_countries
108. base_aareas_schema_administrative_area_administrative_area_type
109. book_book_genre
110. people_person_gender
111. book_book_subject_works
112. meteorology_tropical_cyclone_strongest_storm_of
113. location_statistical_region_external_debt_stock
114. sports_multi_event_tournament_athletic_performances
115. measurement_unit_dated_kgoe_source
116. freebase_type_profile_strict_included_types
117. base_aareas_schema_administrative_area_type_iso_country
118. government_governmental_body_body_this_is_a_component_of
119. location_statistical_region_gdp_real
120. location_statistical_region_official_development_assistance
121. location_statistical_region_military_expenditure_percent_gdp
122. location_location_partially_containedby
123. language_human_language_countries_spoken_in
124. location_statistical_region_child_labor_percent
125. government_government_office_category_offices
126. rdf_schema_domain
127. location_statistical_region_merchandise_trade_percent_of_gdp
128. common_resource_annotations
129. location_country_form_of_government
130. base_athletics_athletics_championships_competition_athlete_relationship_country
131. base_athletics_track_and_field_athlete_championship_events_competed_in
132. base_popstra_vacation_choice_location
133. olympics_olympic_athlete_affiliation_olympics
134. location_statistical_region_gni_in_ppp_dollars
135. olympics_olympic_athlete_country
136. aviation_airport_hub_for
137. aviation_airport_serves
138. symbols_flag_use_flag
139. base_locations_continents_planet
140. time_event_includes_event
141. travel_tourist_attraction_near_travel_destination
142. sports_competitor_country_relationship_country
143. meteorology_tropical_cyclone_category
144. freebase_valuenotation_has_value
145. location_location_time_zones
146. government_governmental_jurisdiction_agencies
147. location_capital_of_administrative_division_capital_of
148. type_property_expected_type
149. olympics_olympic_athlete_medals_won
150. location_administrative_division_capital_relationship_administrative_division
151. sports_competitor_competition_relationship_medal
152. symbols_namesake_named_after
153. organization_organization_founders
154. time_event_locations
155. location_statistical_region_electricity_consumption_per_capita
156. rdf_schema_range
157. base_athletics_athletics_country_championships_athletes_performances
158. location_statistical_region_brain_drain_percent
159. language_human_language_region
160. food_beer_brewery_brand
161. royalty_monarch_kingdom
162. base_athletics_athletics_championships_competition_athlete_relationship_athlete_s
163. location_location_partially_contained_by
164. location_statistical_region_internet_users_percent_population
165. common_topic_article
166. olympics_olympic_games_participating_countries
167. sports_sport_teams
168. sports_competitor_competition_relationship_competitors
169. film_film_language
170. location_statistical_region_energy_use_per_capita
171. sports_multi_event_tournament_competitions
172. geography_river_origin
173. common_webpage_category
174. location_country_languages_spoken
175. royalty_kingdom_rulers
176. sports_competitor_competition_relationship_team
177. base_aareas_schema_administrative_area_administrative_children
178. sports_multi_event_tournament_participating_countries
179. government_national_anthem_of_a_country_country
180. finance_currency_countries_used
181. measurement_unit_adjusted_money_value_source
182. location_statistical_region_gdp_growth_rate
183. freebase_valuenotation_has_no_value
184. base_locations_continents_countries_within
185. organization_organization_membership_organization
186. location_location_people_born_here
187. fictional_universe_fictional_character_gender
188. location_administrative_division_first_level_division_of
189. base_athletics_athletics_championships_competition_athlete_relationship_championships
190. measurement_unit_dated_kilowatt_hour_source
191. travel_travel_destination_tourist_attractions
192. location_statistical_region_health_expenditure_as_percent_of_gdp
193. location_statistical_region_literacy_rate
194. olympics_olympic_medal_honor_olympics
195. olympics_olympic_event_competition_medalists
196. organization_organization_member_member_of
197. base_athletics_athletics_championships_competition_athlete_relationship_medal
198. fictional_universe_fictional_character_place_of_birth
199. base_athletics_athletics_championships_competition_competitors
200. sports_multi_event_tournament_competitors
201. location_statistical_region_co2_emissions_per_capita
202. location_country_first_level_divisions
203. common_webpage_topic
204. location_statistical_region_debt_service_as_percent_of_trade_volume
205. base_popstra_location_vacationers
206. common_image_appears_in_topic_gallery
207. people_deceased_person_place_of_death
208. sports_tournament_event_competitor_country
209. measurement_unit_dated_money_value_source
210. base_ontologies_ontology_instance_equivalent_instances
211. location_country_currency_formerly_used
212. government_governmental_jurisdiction_government_bodies
213. type_type_properties
214. aviation_airport_focus_city_for
215. symbols_flag_referent_flag
216. base_schemastaging_context_name_pronunciation
217. freebase_type_hints_included_types
218. location_statistical_region_time_required_to_start_a_business
219. location_location_contains
220. olympics_olympic_athlete_affiliation_athlete
221. olympics_olympic_event_competition_olympic_games_contested
222. location_statistical_region_high_tech_as_percent_of_manufactured_exports
223. location_location_events
224. sports_sport_country_multi_event_tournaments_participated_in
225. business_employer_employees
226. measurement_unit_dated_money_value_currency
227. location_statistical_region_renewable_freshwater_per_capita
228. common_topic_image
229. location_statistical_region_market_cap_of_listed_companies_as_percent_of_gdp
230. location_location_partiallycontains
231. base_athletics_athletics_medal_medal_winners
232. location_country_capital
233. sports_competitor_country_relationship_tournament
234. common_image_size
235. olympics_olympic_participating_country_athletes
236. olympics_olympic_athlete_affiliation_country
237. location_country_internet_tld
238. sports_sports_team_location_teams
239. people_person_place_of_birth
240. olympics_olympic_participating_country_medals_won
241. organization_organization_geographic_scope
242. music_composition_language
243. film_film_featured_film_locations
244. measurement_unit_adjusted_money_value_adjustment_currency
245. base_locations_planets_continents_within
246. location_statistical_region_consumer_price_index
247. sports_tournament_event_competition_tournament
248. location_statistical_region_trade_balance_as_percent_of_gdp
249. food_beer_country_region_beers_from_here
250. measurement_unit_dated_metric_ton_source
251. government_governmental_body_component_bodies
252. olympics_olympic_participating_country_olympics_participated_in
253. book_author_works_written
254. location_statistical_region_long_term_unemployment_rate
255. finance_currency_countries_formerly_used

Graph indentifier:
Add the identification label "webqsp" to the entities in the cypher query, for example, ":webqsp:node" for the entities in the graph.
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

Questions can be about node inquiries or relations between nodes.

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

User questions are about node inquiries which involve multi-hop relation paths.

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

4. Return the unique ids of retrieved entities and set the label of the result column as "id", using `RETURN DISTINCT node.id as id` in the cypher query.

5. Use "LIMIT 20" to limit the number of results returned in the cypher query.

6. return the cypher query in the following format:
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

5. Always use single "MATCH path =" clause for the whole cypher query, starting from one input entity to another and captures the whole path.

6. Return the results in path format:
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

5. Always use single "MATCH path =" clause for the whole cypher query, starting from one input entity to another and captures the whole path.

6. Return the results in path format:
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

5. Always use single "MATCH path =" clause for the whole cypher query, starting from one input entity to another and captures the whole path.

6. Return the results in path format:
```cypher
RETURN path
LIMIT 10
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

You should return two columns "source" and "target" using the "id" property, indicating relations consisting of the source and target entity IDs that are required to answer the question: (1) source node entity IDs and (2) target node entity IDs.

If you don't have adequate information to give trustful reasoning paths, just say so. Do not make anything up.

Do not make up any information where the supporting evidence for it is not provided.

---Graph Schema---

{graph_schema}

---Notes---

1. Please carefully think about the query structure and make sure the query is **correct** and **efficient** to execute. Do not forget to assign a variable name before retrieving the attributes.

2. Add the identification label in the generated cypher query as instructed in the graph schema section to ensure the cypher query inquires about the right graph.

3. Use the "id" property to identify the entities in the graph, rather than names. Users will provide the ids of the entities along with the questions.

4. Only return two columns "souce" and "target" which are the entity IDs of the required relations, using "id" property. Do not involve any other columns or attributes.

5. return the cypher query in the following format:
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
When given a question, analyze it based on the definitions above. If the question belongs to any of the four types, then return **only a single number** corresponding to the type of the question.
However, note that not every question can be directly classified into the above four classes, where the question is nested and asks about different types of relations, in which case you should return **-1**.
For example, when the question asks about the relationship between two entities or inquiry about general information about some entities, but the entities need to be determined by another query embeded in the overall question, it is a nested question and should be classified as -1.

Especially, nested questions can only be in the form where the overall quesiton is one of <s,*,*>, <s,*,o>, <s,p,o> question nested with <s,p,*> question.
If the question reveals a chain of specific relations (multi-hop predicates) from one specific entity (subject), it is not a nested question and should be classified as <s,p,*> question and output 1.
In the case of multiple sequential <s,p,*> sub-questions with different subjects, you can merge them into one single <s,p,*> question and classify the overall question as 1 for <s,p,*>, as consecutive <s,p,*> questions can always be merged into one single <s,p,*> question with multiple predicates and subjects.

Some concrete example:
- "Provide some concrete information about the academic collaborators of the scholar 'L. Foldy'.": 1. the first step is a <s,p,*> question, finding the collaborators of 'L. Foldy'. 2. the second step is a <s,*,*> question: gather general and broad information for the collaborators.
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
- For <s,*,*> and <s,p,*> question, you can merge several questions into one question. For example two <s,*,*> questions, "Tell me about 'A'." and "Tell me about 'B'." can be merged into "Tell me about 'A' and 'B'.". And you should give both the id mapping for 'A' and 'B'. Same for <s,p,*> questions, for example "Who is the collaborator of both 'A' and 'B'?".
- For other two types of questions <s,*,o> and <s,p,o>, you need to generate a seperate concrete question for each entity. For example, "What is the relation between 'L. Foldy' and 'A'?" and "What is the relation between 'L. Foldy' and 'B'?" should be generated as two separate questions. And you should give a seperate id mapping for each concrete question.

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


# PROMPTS["cypher_only_query_free_form_output"] = """---Role---
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
