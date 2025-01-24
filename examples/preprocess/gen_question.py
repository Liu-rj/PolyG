import random
import jsonlines
import argparse
import pickle
import os
import time
import networkx as nx
from typing import List
from neo4j import GraphDatabase


neo4j_config = {
    "neo4j_url": os.environ.get("NEO4J_URL", "neo4j://localhost:7687"),
    "neo4j_auth": (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "12345678"),
    ),
}


single_entity_concrete_template = {
    "physics": {
        "author": {
            "What paper have the author '{}' published?": {
                "cypher": """
            MATCH (author:physics:author {{id: '{}'}})
            -[:paper]->(paper:physics:paper)
            RETURN paper.name as name
            """,
                "hops": 1,
            },
            "What are the academic collaborators of '{}'?": {
                "cypher": """
            MATCH (author:physics:author {{id: '{}'}})
            -[:paper]->(paper:physics:paper)
            -[:author]->(collaborator:physics:author)
            WHERE collaborator <> author
            RETURN DISTINCT collaborator.name AS name
            """,
                "hops": 2,
            },
            "What venues have the author '{}' published in?": {
                "cypher": """
            MATCH (author:physics:author {{id: '{}'}})
            -[:paper]->(paper:physics:paper)
            -[:venue]->(venue:physics:venue)
            RETURN DISTINCT venue.name AS name
            """,
                "hops": 2,
            },
        },
        "paper": {
            "Who are the authors of the paper '{}'?": {
                "cypher": """
            MATCH (p:physics:paper {{id: '{}'}})
            -[:author]->(a:physics:author)
            RETURN DISTINCT a.name AS name
            """,
                "hops": 1,
            },
            "Where is the paper '{}' published?": {
                "cypher": """
            MATCH (p:Physics:paper {{id: '{}'}})
            -[:venue]->(v:Physics:venue)
            RETURN DISTINCT v.name AS name
            """,
                "hops": 1,
            },
            "Who are the academic collaborators of the author who writes the paper '{}'?": {
                "cypher": """
            MATCH (p:physics:paper {{id: '{}'}})
            MATCH (p)-[:author]->(a:physics:author)
            MATCH (a)-[:paper]->(otherPaper:physics:paper)
            MATCH (otherPaper)-[:author]->(coAuthor:physics:author)
            WHERE coAuthor <> a
            RETURN DISTINCT coAuthor.name AS name
            """,
                "hops": 3,
            },
            "What venues have the author of the paper '{}' published in?": {
                "cypher": """
            MATCH (p:physics:paper {{id: '{}'}})
            -[:author]->(a:physics:author)
            -[:paper]->(other_p:physics:paper)
            -[:venue]->(v:physics:venue)
            RETURN DISTINCT v.name AS name
            """,
                "hops": 3,
            },
            "What venues have the academic collaborators of the author who writes the paper '{}' published in?": {
                "cypher": """
            MATCH (start_paper:physics:paper {{id: '{}'}})
            -[:author]->(author:physics:author)
            -[:paper]->(collab_paper:physics:paper)
            -[:author]->(collaborator:physics:author)
            WHERE collaborator <> author
            MATCH (collaborator)-[:paper]->(pub:physics:paper)
            -[:venue]->(venue:physics:venue)
            RETURN DISTINCT venue.name AS name
            """,
                "hops": 5,
            },
        },
    },
    "amazon": {
        "item": {
            "What is the brand of the item '{}'?": {
                "cypher_template": """
            MATCH (a1:amazon:item)-[:brand]->(brand1:amazon:brand)
            RETURN DISTINCT a1.name as name, a1.id as id LIMIT 10000
            """,
                "cypher": """
            MATCH (:amazon:item {{id: '{}'}})
            -[:brand]->(brand1:amazon:brand)
            RETURN DISTINCT brand1.name as name
            """,
                "hops": 1,
            },
            "What are the brands of the items that are also bought after viewing the item '{}'?": {
                "cypher_template": """
            MATCH (a1:amazon:item)-[:buy_after_viewing_item]->(bought_item:amazon:item)-[:brand]->(brand1:amazon:brand)
            RETURN DISTINCT a1.name as name, a1.id as id LIMIT 10000
            """,
                "cypher": """
            MATCH (:amazon:item {{id: '{}'}})-[:buy_after_viewing_item]->(bought_item:amazon:item)-[:brand]->(brand1:amazon:brand)
            RETURN DISTINCT brand1.name as name
            """,
                "hops": 2,
            },
            "What are the items that are also viewed when viewing items of the brand owning the item '{}'?": {
                "cypher_template": """
            MATCH (start:amazon:item)
            -[:brand]->(brand:amazon:brand)
            -[:item]->(other_items:amazon:item)
            -[:also_viewed_item]->(also_viewed:amazon:item)
            RETURN DISTINCT start.name as name, start.id as id LIMIT 10000
            """,
                "cypher": """
            MATCH (start:amazon:item {{id: '{}'}})
            -[:brand]->(brand:amazon:brand)
            -[:item]->(other_items:amazon:item)
            -[:also_viewed_item]->(also_viewed:amazon:item)
            RETURN DISTINCT also_viewed.name as name
            """,
                "hops": 3,
            },
            "What are the brands of the items that are also bought with items of the brand owning the item '{}'?": {
                "cypher_template": """
            MATCH (start:amazon:item)
            -[:brand]->(brand:amazon:brand)
            -[:item]->(same_brand_items:amazon:item)
            -[:also_bought_item]->(also_bought_items:amazon:item)
            -[:brand]->(result_brands:amazon:brand)
            RETURN DISTINCT start.name as name, start.id as id LIMIT 1000
            """,
                "cypher": """
            MATCH (start:amazon:item {{id: '{}'}})
            -[:brand]->(brand:amazon:brand)
            -[:item]->(same_brand_items:amazon:item)
            -[:also_bought_item]->(also_bought_items:amazon:item)
            -[:brand]->(result_brands:amazon:brand)
            RETURN DISTINCT result_brands.name as name
            """,
                "hops": 4,
            },
        },
        "brand": {
            "What are the items of the brand '{}'?": {
                "cypher_template": """
            MATCH (b:amazon:brand)-[:item]->(i:amazon:item)
            RETURN DISTINCT b.name as name, b.id as id LIMIT 10000
            """,
                "cypher": """
            MATCH (b:amazon:brand {{id: '{}'}})-[:item]->(i:amazon:item)
            RETURN DISTINCT i.name as name
            """,
                "hops": 1,
            },
            "What are the items that are bought together with items of the brand '{}'?": {
                "cypher_template": """
            MATCH (b1:amazon:brand)-[:item]->(:amazon:item)-[:bought_together_item]->(bought_together_items:amazon:item)
            RETURN DISTINCT b1.name as name, b1.id as id LIMIT 10000
            """,
                "cypher": """
            MATCH (:amazon:brand {{id: '{}'}})-[:item]->(:amazon:item)-[:bought_together_item]->(bought_together_items:amazon:item)
            RETURN DISTINCT bought_together_items.name as name
            """,
                "hops": 2,
            },
            "What are the brands of the items that are also bought with items of the brand '{}'?": {
                "cypher_template": """
            MATCH (start_brand:amazon:brand)
            -[:item]->(brand_item:amazon:item)
            -[:also_bought_item]->(bought_item:amazon:item)
            -[:brand]->(result_brand:amazon:brand)
            RETURN DISTINCT start_brand.name as name, start_brand.id as id LIMIT 10000
            """,
                "cypher": """
            MATCH (start_brand:amazon:brand {{id: '{}'}})
            -[:item]->(brand_item:amazon:item)
            -[:also_bought_item]->(bought_item:amazon:item)
            -[:brand]->(result_brand:amazon:brand)
            RETURN DISTINCT result_brand.name as name
            """,
                "hops": 3,
            },
            "What items does the brands of the items that are also viewed together with items of the brand '{}' have?": {
                "cypher_template": """
            MATCH (brandA:amazon:brand)
            -[:item]->(itemA:amazon:item)
            -[:also_viewed_item]->(alsoViewedItem:amazon:item)
            -[:brand]->(otherBrand:amazon:brand)
            -[:item]->(resultItem:amazon:item)
            RETURN DISTINCT brandA.name as name, brandA.id as id LIMIT 10000
            """,
                "cypher": """
            MATCH (brandA:amazon:brand {{id: '{}'}})
            -[:item]->(itemA:amazon:item)
            -[:also_viewed_item]->(alsoViewedItem:amazon:item)
            -[:brand]->(otherBrand:amazon:brand)
            -[:item]->(resultItem:amazon:item)
            RETURN DISTINCT resultItem.name as name
            """,
                "hops": 4,
            },
        },
    },
    "goodreads": {
        "book": {
            "Who are the authors of the book '{}'?": {
                "cypher": """
            MATCH (book:goodreads:book {{id: '{}'}})
            -[:author]->(author:goodreads:author)
            RETURN DISTINCT author.name as name
            """,
                "hops": 1,
            },
            "What series have the author of the book '{}' published?": {
                "cypher": """
            MATCH (book:goodreads:book {{id: '{}'}})
            MATCH (book)-[:author]->(author:goodreads:author)
            MATCH (author)-[:book]->(other_books:goodreads:book)
            MATCH (other_books)-[:series]->(series:goodreads:series)
            RETURN DISTINCT series.name as name
            """,
                "hops": 3,
            },
        },
        "author": {
            "What books has the author '{}' published?": {
                "cypher": """
            MATCH (author:goodreads:author {{id: '{}'}})
            -[:book]->(book:goodreads:book)
            RETURN DISTINCT book.name as name
            """,
                "hops": 1,
            },
            "What books have the collaborators of the author '{}' published?": {
                "cypher": """
            MATCH (author1:goodreads:author {{id: '{}'}})-[:book]->(book1:goodreads:book)
            MATCH (book1)-[:author]->(coauthor:goodreads:author)
            WHERE coauthor <> author1
            MATCH (coauthor)-[:book]->(other_book:goodreads:book)
            RETURN DISTINCT other_book.name as name
            """,
                "hops": 3,
            },
            "What are the series published by the publishers that have published books of the author '{}'?": {
                "cypher": """
            MATCH (author:goodreads:author {{id: '{}'}})
            -[:book]->(authorBook:goodreads:book)
            -[:publisher]->(publisher:goodreads:publisher)
            -[:book]->(publisherBook:goodreads:book)
            -[:series]->(series:goodreads:series)
            RETURN DISTINCT series.name as name
            """,
                "hops": 4,
            },
        },
        "publisher": {
            "What are the authors of the books published by the publisher '{}'?": {
                "cypher": """
            MATCH (:goodreads:publisher {{id: '{}'}})
            -[:book]->(b:goodreads:book)-[:author]->(a:goodreads:author)
            RETURN DISTINCT a.name as name
            """,
                "hops": 2,
            },
        },
        "series": {
            "Where does the books of the series '{}' published in?": {
                "cypher": """
            MATCH (s:goodreads:series {{id: '{}'}})
            MATCH (s)-[:book]->(b:goodreads:book)-[:publisher]->(p:goodreads:publisher)
            RETURN DISTINCT p.name as name
            """,
                "hops": 2,
            },
            "What are the authors of the books that are published by the publishers that have published books of the series '{}'?": {
                "cypher": """
            MATCH (s:goodreads:series {{id: '{}'}})
            -[:book]->(book1:goodreads:book)
            -[:publisher]->(p:goodreads:publisher)
            -[:book]->(book2:goodreads:book)
            -[:author]->(a:goodreads:author)
            RETURN DISTINCT a.name as name
            """,
                "hops": 4,
            },
        },
    },
}


multi_entity_concrete_template = {
    "physics": {
        "Have the author '{}' cited or been cited by the work of the author '{}' and what are those works?": {
            "cypher_template": """
            MATCH (author1:physics:author)
            -[:paper]->(paper1:physics:paper)
            -[:reference|cited_by]->(paper2:physics:paper)
            -[:author]->(author2:physics:author)
            WHERE author1 <> author2
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "cypher": """
            MATCH path = (author1:physics:author {{id: '{}'}})
            -[:paper]->(paper1:physics:paper)
            -[:reference|cited_by]->(paper2:physics:paper)
            -[:author]->(author2:physics:author {{id: '{}'}})
            RETURN path LIMIT 10
            """,
            "hops": 3,
        },
        "Have authors '{}' and '{}' both collaborated with some other authors and who are they?": {
            "cypher_template": """
            MATCH (author1:physics:author)
            -[:paper]->(paper1:physics:paper)
            -[:author]->(collaborator:physics:author)
            -[:paper]->(paper2:physics:paper)
            -[:author]->(author2:physics:author)
            WHERE author1 <> collaborator AND author2 <> collaborator
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "cypher": """
            MATCH path = (author1:physics:author {{id: '{}'}})
            -[:paper]->(paper1:physics:paper)
            -[:author]->(collaborator:physics:author)
            -[:paper]->(paper2:physics:paper)
            -[:author]->(author2:physics:author {{id: '{}'}})
            WHERE author1 <> collaborator AND author2 <> collaborator
            RETURN path LIMIT 10
            """,
            "hops": 4,
        },
        "Have the authors '{}' and '{}' ever published papers in the same venues? If so, tell me some examples.": {
            "cypher_template": """
            MATCH (author1:physics:author)
            -[:paper]->(paper1:physics:paper)
            -[:venue]->(venue:physics:venue)
            -[:paper]->(paper2:physics:paper)
            -[:author]->(author2:physics:author)
            WHERE author1 <> author2
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "cypher": """
            MATCH path = (author1:physics:author {{id: '{}'}})
            -[:paper]->(paper1:physics:paper)
            -[:venue]->(venue:physics:venue)
            -[:paper]->(paper2:physics:paper)
            -[:author]->(author2:physics:author {{id: '{}'}})
            RETURN path LIMIT 10
            """,
            "hops": 4,
        },
        "Do the venues '{}' and '{}' have the same authors publishing work in both of them and who are they?": {
            "cypher_template": """
            MATCH (v1:physics:venue)<-[:venue]-(:physics:paper)<-[:paper]-(a:physics:author)-[:paper]->(:physics:paper)-[:venue]->(v2:physics:venue)
            RETURN v1.name AS name1, v1.id AS id1, v2.name AS name2, v2.id AS id2
            """,
            "cypher": """
            MATCH path = (v1:physics:venue {{id: '{}'}})
            -[:paper]->(:physics:paper)
            -[:author]->(a:physics:author)
            -[:paper]->(:physics:paper)
            -[:venue]->(v2:physics:venue {{id: '{}'}})
            RETURN path LIMIT 10
            """,
            "hops": 4,
        },
        # NGW"What is the collaboration relationship between the authors of the paper '{}' and '{}'?": {
        #     "cypher_template": """
        #     MATCH (paper1:physics:paper)-[:author]->(author1:physics:author)
        #     -[:paper]->(sharedPaper:physics:paper)<-[:paper]-(author2:physics:author)
        #     <-[:author]-(paper2:physics:paper)
        #     WHERE author1 <> author2
        #     RETURN paper1.name AS name1, paper1.id AS id1, paper2.name AS name2, paper2.id AS id2
        #     """,
        #     "cypher": """
        #     MATCH path = (paper1:physics:paper {{id: '{}'}})
        #     -[:author]->(author1:physics:author)
        #     -[:paper]->(sharedPaper:physics:paper)
        #     -[:author]->(author2:physics:author)
        #     -[:paper]->(paper2:physics:paper {{id: '{}'}})
        #     WHERE author1 <> author2
        #     RETURN path LIMIT 10
        #     """,
        #     "hops": 4,
        # },
    },
    "amazon": {
        # NGW"Are the items '{}' and '{}' both also bought with items of some other brands? If so, tell me about those brands and their items.": {
        #     "cypher_template": """
        #     MATCH (itemA:amazon:item)-[:also_bought_item]->(viewedItemA:amazon:item)-[:brand]->(brandA:amazon:brand)<-[:brand]-(viewedItemB:amazon:item)<-[:also_bought_item]-(itemB:amazon:item)
        #     WHERE itemA <> itemB
        #     RETURN itemA.name AS name1, itemA.id AS id1, itemB.name AS name2, itemB.id AS id2
        #     """,
        #     "cypher": """
        #     MATCH path = (itemA:amazon:item {{id: '{}'}})-[:also_bought_item]->(viewedItemA:amazon:item)-[:brand]->(brandA:amazon:brand)-[:item]->(viewedItemB:amazon:item)-[:also_bought_item]->(itemB:amazon:item {{id: '{}'}})
        #     RETURN path LIMIT 10
        #     """,
        #     "hops": 4,
        # },
        # NGW"Are the items '{}' and '{}' both also viewed with items of some other brands and what are those items?": {
        #     "cypher_template": """
        #     MATCH (item1:amazon:item)-[:also_viewed_item]->(also_viewed1:amazon:item)-[:brand]->(brand:amazon:brand)<-[:brand]-(also_viewed2:amazon:item)<-[:also_viewed_item]-(item2:amazon:item)
        #     WHERE item1 <> item2
        #     RETURN item1.name AS name1, item1.id AS id1, item2.name AS name2, item2.id AS id2
        #     """,
        #     "cypher": """
        #     MATCH path = (item1:amazon:item {{id: '{}'}})-[:also_viewed_item]->(also_viewed1:amazon:item)-[:brand]->(brand:amazon:brand)-[:item]->(also_viewed2:amazon:item)-[:also_viewed_item]->(item2:amazon:item {{id: '{}'}})
        #     RETURN path LIMIT 10
        #     """,
        #     "hops": 4,
        # },
        "Have the items of the brands '{}' and '{}' ever both been also bought with some other items, and if so, what are those items?": {
            "cypher_template": """
            MATCH (brandA:amazon:brand)-[:item]->(itemA:amazon:item)-[:also_bought_item]->(viewedItem:amazon:item)<-[:also_bought_item]-(itemB:amazon:item)<-[:item]-(brandB:amazon:brand)
            WHERE brandA <> brandB
            RETURN brandA.name AS name1, brandA.id AS id1, brandB.name AS name2, brandB.id AS id2
            """,
            "cypher": """
            MATCH path = (brandA:amazon:brand {{id: '{}'}})-[:item]->(itemA:amazon:item)-[:also_bought_item]->(viewedItem:amazon:item)-[:also_bought_item]->(itemB:amazon:item)-[:brand]->(brandB:amazon:brand {{id: '{}'}})
            RETURN path LIMIT 10
            """,
            "hops": 4,
        },
        "Have the items of the brands '{}' and '{}' ever both been bought after viewing some other items, and if so, what are those items?": {
            "cypher_template": """
            MATCH (b1:amazon:brand)-[:item]->(itemA:amazon:item)<-[:buy_after_viewing_item]-(commonItem:amazon:item)-[:buy_after_viewing_item]->(itemB:amazon:item)<-[:item]-(b2:amazon:brand)
            WHERE b1 <> b2
            RETURN b1.name AS name1, b1.id AS id1, b2.name AS name2, b2.id AS id2
            """,
            "cypher": """
            MATCH path = (b1:amazon:brand {{id: '{}'}})-[:item]->(itemA:amazon:item)<-[:buy_after_viewing_item]-(commonItem:amazon:item)-[:buy_after_viewing_item]->(itemB:amazon:item)-[:brand]->(b2:amazon:brand {{id: '{}'}})
            RETURN path LIMIT 10
            """,
            "hops": 4,
        },
        "Have the items of the brands '{}' and '{}' ever been viewed together with some other items, and if so, what are those items?": {
            "cypher_template": """
            MATCH (brandA:amazon:brand)-[:item]->(itemA:amazon:item)-[:also_viewed_item]->(viewedItem:amazon:item)<-[:also_viewed_item]-(itemB:amazon:item)<-[:item]-(brandB:amazon:brand)
            WHERE brandA <> brandB
            RETURN brandA.name AS name1, brandA.id AS id1, brandB.name AS name2, brandB.id AS id2
            """,
            "cypher": """
            MATCH path = (brandA:amazon:brand {{id: '{}'}})-[:item]->(itemA:amazon:item)-[:also_viewed_item]->(viewedItem:amazon:item)-[:also_viewed_item]->(itemB:amazon:item)-[:brand]->(brandB:amazon:brand {{id: '{}'}})
            RETURN path LIMIT 10
            """,
            "hops": 4,
        },
        "Have the items of the brands '{}' and '{}' ever been bought together with some other items, and if so, what are those items?": {
            "cypher_template": """
            MATCH (b1:amazon:brand)-[:item]->(itemA:amazon:item)-[:bought_together_item]->(otherItem:amazon:item)<-[:bought_together_item]-(itemB:amazon:item)<-[:item]-(b2:amazon:brand)
            WHERE b1 <> b2
            RETURN b1.name AS name1, b1.id AS id1, b2.name AS name2, b2.id AS id2
            """,
            "cypher": """
            MATCH path = (b1:amazon:brand {{id: '{}'}})-[:item]->(itemA:amazon:item)-[:bought_together_item]->(otherItem:amazon:item)-[:bought_together_item]->(itemB:amazon:item)-[:brand]->(b2:amazon:brand {{id: '{}'}})
            RETURN path LIMIT 10
            """,
            "hops": 4,
        },
    },
    "goodreads": {
        # NGW"What is the relationship between authors '{}' and '{}' regarding collaborated books?": {
        #     "cypher_template": """
        #     MATCH path = (author1:goodreads:author)-[:book]->(book:goodreads:book)<-[:book]-(author2:goodreads:author)
        #     WHERE author1 <> author2
        #     RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
        #     """,
        #     "cypher": """
        #     MATCH path = (author1:goodreads:author {{id: '{}'}})-[:book]->(book:goodreads:book)-[:author]->(author2:goodreads:author {{id: '{}'}})
        #     RETURN path LIMIT 10
        #     """,
        #     "hops": 2,
        # },
        # NGW"Have the authors '{}' and '{}' published books that belongs to the same series and what are they?": {
        #     "cypher_template": """
        #     MATCH path = (author1:goodreads:author)-[:book]->(book1:goodreads:book)-[:series]->(series:goodreads:series)<-[:series]-(book2:goodreads:book)<-[:book]-(author2:goodreads:author)
        #     WHERE author1 <> author2
        #     RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
        #     """,
        #     "cypher": """
        #     MATCH path = (author1:goodreads:author {{id: '{}'}})-[:book]->(book1:goodreads:book)-[:series]->(series:goodreads:series)-[:book]->(book2:goodreads:book)-[:author]->(author2:goodreads:author {{id: '{}'}})
        #     RETURN path LIMIT 10
        #     """,
        #     "hops": 4,
        # },
        "Have the authors '{}' and '{}' ever published books in the same publishers? If so, tell me some examples.": {
            "cypher_template": """
            MATCH path = (author1:goodreads:author)-[:book]->(book1:goodreads:book)-[:publisher]->(publisher:goodreads:publisher)<-[:publisher]-(book2:goodreads:book)<-[:book]-(author2:goodreads:author)
            WHERE author1 <> author2
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "cypher": """
            MATCH path = (author1:goodreads:author {{id: '{}'}})-[:book]->(book1:goodreads:book)-[:publisher]->(publisher:goodreads:publisher)-[:book]->(book2:goodreads:book)-[:author]->(author2:goodreads:author {{id: '{}'}})
            RETURN path LIMIT 10
            """,
            "hops": 4,
        },
        "Do the publishers '{}' and '{}' have any authors publishing books in both of them and what are the publications and authors?": {
            "cypher_template": """
            MATCH path = (publisher1:goodreads:publisher)-[:book]->(book1:goodreads:book)-[:author]->(author:goodreads:author)-[:book]->(book2:goodreads:book)-[:publisher]->(publisher2:goodreads:publisher)
            WHERE publisher1 <> publisher2
            RETURN publisher1.name AS name1, publisher1.id AS id1, publisher2.name AS name2, publisher2.id AS id2
            """,
            "cypher": """
            MATCH path = (publisher1:goodreads:publisher {{id: '{}'}})-[:book]->(book1:goodreads:book)-[:author]->(author:goodreads:author)-[:book]->(book2:goodreads:book)-[:publisher]->(publisher2:goodreads:publisher {{id: '{}'}})
            RETURN path LIMIT 10
            """,
            "hops": 4,
        },
        "Do the publishers '{}' and '{}' have books that belong to the same series, and if so, what are those books?": {
            "cypher_template": """
            MATCH (p1:goodreads:publisher)-[:book]->(:goodreads:book)-[:series]->(:goodreads:series)<-[:series]-(:goodreads:book)<-[:book]-(p2:goodreads:publisher)
            where p1 <> p2
            RETURN p1.name AS name1, p1.id AS id1, p2.name AS name2, p2.id AS id2
            """,
            "cypher": """
            MATCH path = (:goodreads:publisher {{id: '{}'}})-[:book]->(:goodreads:book)-[:series]->(:goodreads:series)-[:book]->(:goodreads:book)-[:publisher]->(:goodreads:publisher {{id: '{}'}})
            RETURN path LIMIT 10
            """,
            "hops": 4,
        },
        "Do the series '{}' and '{}' contain books that are published by the same publisher? If so, tell me about them.": {
            "cypher_template": """
            MATCH (s1:goodreads:series)-[:book]->(:goodreads:book)-[:publisher]->(:goodreads:publisher)-[:book]->(:goodreads:book)-[:series]->(s2:goodreads:series)
            RETURN s1.name AS name1, s1.id AS id1, s2.name AS name2, s2.id AS id2
            """,
            "cypher": """
            MATCH path = (:goodreads:series {{id: '{}'}})-[:book]->(:goodreads:book)-[:publisher]->(:goodreads:publisher)-[:book]->(:goodreads:book)-[:series]->(:goodreads:series {{id: '{}'}})
            RETURN path LIMIT 10
            """,
            "hops": 4,
        },
        # NGW"Are there authors who have published books in both the series '{}' and '{}' and what are they?": {
        #     "cypher_template": """
        #     MATCH (series1:goodreads:series)-[:book]->(book1:goodreads:book)-[:author]->(author:goodreads:author)-[:book]->(book2:goodreads:book)-[:series]->(series2:goodreads:series)
        #     RETURN series1.name AS name1, series1.id AS id1, series2.name AS name2, series2.id AS id2
        #     """,
        #     "cypher": """
        #     MATCH path = (series1:goodreads:series {{id: '{}'}})-[:book]->(book1:goodreads:book)-[:author]->(author:goodreads:author)-[:book]->(book2:goodreads:book)-[:series]->(series2:goodreads:series {{id: '{}'}})
        #     RETURN path LIMIT 10
        #     """,
        #     "hops": 4,
        # },
    },
}


def gen_single_entity_abstract(graph: nx.Graph, n: int, output_path: str):
    all_nodes = list(graph.nodes())
    questions = []
    i = 0
    name_set = {""}
    while i < n:
        node = random.choice(all_nodes)
        node_data = graph.nodes[node]
        if node_data["name"] in name_set:
            continue
        # if it has no neighbors, skip
        if graph.degree(node) < 10:
            continue
        question = f"Tell me about '{node_data['name']}'."
        q_entity = {
            "qid": i,
            "question": question,
            "entity": {node_data["name"]: node},
            "type": "single_entity_abstract",
            "hops": 1,
            "answer": "N/A",
        }
        questions.append(q_entity)
        name_set.add(node_data["name"])
        print(q_entity)
        i += 1
    with jsonlines.open(output_path, "w") as writer:
        for row in questions:
            writer.write(row)


def choose_random_node(neo4j_driver, namespace):
    with neo4j_driver.session() as session:
        result = session.run(
            f"""
            MATCH (n:{namespace})
            RETURN n.id as id, n.name as name
            ORDER BY rand()
            LIMIT 1
            """
        )
        node = result.single()
        return node["name"], node["id"]


def gen_single_entity_concrete(
    graph: nx.Graph, n: int, graph_name: str, output_path: str
):
    neo4j_url = neo4j_config["neo4j_url"]
    neo4j_auth = neo4j_config["neo4j_auth"]
    driver = GraphDatabase.driver(
        neo4j_url,
        auth=neo4j_auth,
        max_connection_pool_size=100,
        connection_timeout=60,
    )
    q_templates = single_entity_concrete_template[graph_name]
    questions = []
    for node_type, templates in q_templates.items():
        for q, content in templates.items():
            q_cypher, n_hop = content["cypher"], content["hops"]

            candidates = []
            if "cypher_template" in content:
                print("Using cypher template")
                with driver.session() as session:
                    results = session.run(content["cypher_template"])
                    for record in results:
                        candidates.append((record["name"], record["id"]))

            i = 0
            selected_nodes = set()
            while i < n:
                if len(candidates) > 0:
                    print("Using candidates")
                    node_name, node = candidates.pop(0)
                    if node_name in selected_nodes:
                        continue
                else:
                    print("Using random node")
                    node_name, node = choose_random_node(
                        driver, f"{graph_name}:{node_type}"
                    )

                result_names = []
                continue_flag = False
                with driver.session() as session:
                    try:
                        with session.begin_transaction(timeout=10) as tx:
                            result = tx.run(q_cypher.format(node))
                            for record in result:
                                result_names.append(record["name"])
                                if len(result_names) > 20:
                                    print(
                                        f"{i}: Too many results, cypher: {q_cypher.format(node)}"
                                    )
                                    continue_flag = True
                                    break
                    except Exception as e:
                        print(f"Query failed: {e}")
                        continue_flag = True
                if continue_flag:
                    continue
                if None in result_names:
                    print(f"{i}: None in results, cypher: {q_cypher.format(node)}")
                    continue
                if len(result_names) == 0:
                    print(f"{i}: empty results, cypher: {q_cypher.format(node)}")
                    continue
                if n_hop > 1 and len(result_names) < 3:
                    print(f"{i}: Too few results, cypher: {q_cypher.format(node)}")
                    continue

                question = q.format(node_name)
                q_entity = {
                    "qid": len(questions),
                    "question": question,
                    "entity": {node_name: node},
                    "type": "single_entity_concrete",
                    "hops": n_hop,
                    "answer": ", ".join(result_names),
                }
                questions.append(q_entity)
                selected_nodes.add(node_name)
                print(q_entity)
                i += 1

    with jsonlines.open(output_path, "w") as writer:
        for row in questions:
            writer.write(row)


def gen_multi_entity_abstract(
    graph: nx.Graph, n: int, hops: List[int], output_path: str
):
    def bfs_with_path_length(graph, start_node, length):
        queue = [(start_node, 0)]
        visited = {graph.nodes.get(start_node)["name"]}

        while queue:
            current_node, current_length = queue.pop(0)
            if current_length == length:
                return current_node
            elif current_length < length:
                for neighbor in graph.neighbors(current_node):
                    name = graph.nodes[neighbor]["name"]
                    if name not in visited:
                        visited.add(name)
                        queue.append((neighbor, current_length + 1))
        return None

    all_nodes = list(graph.nodes())
    questions = []
    for it, n_hop in enumerate(hops):
        i = 0
        while i < n:
            start_node = random.choice(all_nodes)
            end_node = bfs_with_path_length(graph, start_node, n_hop)
            if end_node is None:
                continue
            start_node_data = graph.nodes[start_node]
            end_node_data = graph.nodes[end_node]
            question = f"What is the relationship between '{start_node_data['name']}' and '{end_node_data['name']}'?"
            q_entity = {
                "qid": i + it * n,
                "question": question,
                "entity": {
                    start_node_data["name"]: start_node,
                    end_node_data["name"]: end_node,
                },
                "type": "multi_entity_abstract",
                "hops": n_hop,
                "answer": "N/A",
            }
            questions.append(q_entity)
            print(q_entity)
            i += 1

    with jsonlines.open(output_path, "w") as writer:
        for row in questions:
            writer.write(row)


def all_shortest_paths(
    neo4j_driver, namespace, source: str, target: str
) -> list[list[str]]:
    with neo4j_driver.session() as session:
        result = session.run(
            f"""
            MATCH p = SHORTEST 10 (s:{namespace} {{id: $source_id}})
            -[*]->(t:{namespace} {{id: $target_id}})
            RETURN [n in nodes(p) | n.id] AS path
            """,
            source_id=source,
            target_id=target,
        )

        paths = []
        for record in result:
            node_id = record["path"]
            paths.append(node_id)
        return paths


def gen_multi_entity_concrete(
    graph: nx.Graph, n: int, graph_name: str, output_path: str
):
    neo4j_url = neo4j_config["neo4j_url"]
    neo4j_auth = neo4j_config["neo4j_auth"]
    driver = GraphDatabase.driver(
        neo4j_url,
        auth=neo4j_auth,
        max_connection_pool_size=100,
        connection_timeout=60,
    )
    q_templates = multi_entity_concrete_template[graph_name]
    questions = []
    for it, (q, content) in enumerate(q_templates.items()):
        q_cypher_t, n_hop = content["cypher_template"], content["hops"]
        q_cypher = content["cypher"]

        result_list = []
        name_set = set()
        retry_counts = 0
        with driver.session() as session:
            results = session.run(q_cypher_t)
            for record in results:
                if record["name1"] == record["name2"]:
                    continue
                if "venues '{}' and '{}'" in q:
                    if record["name1"] in name_set and record["name2"] in name_set:
                        continue
                else:
                    if record["name1"] in name_set or record["name2"] in name_set:
                        continue
                print(f"Getting the next record for question: {q}")

                try:
                    # Use a separate session for the nested query
                    with driver.session() as inner_session:
                        with inner_session.begin_transaction(timeout=10) as tx:
                            tic = time.time()
                            record_ret = tx.run(
                                q_cypher.format(record["id1"], record["id2"])
                            )
                            duration = time.time() - tic
                            num_path = 0
                            for r in record_ret:
                                num_path += 1
                            if num_path < 10:
                                print(f"Too few paths: {num_path}, retry")
                                continue
                except Exception as e:
                    print(f"Query failed: {e}")
                    print(f"Query: {q_cypher.format(record['id1'], record['id2'])}")
                    continue

                count = 0
                if n_hop > 2 and retry_counts < 2000:
                    print("Checking overlap with shortest paths")
                    shortest_paths = all_shortest_paths(
                        driver, graph_name, record["id1"], record["id2"]
                    )
                    for path in shortest_paths:
                        if len(path) - 1 >= n_hop:
                            count += 1
                if count >= 5:
                    print("Overlap with shortest paths, retry")
                    retry_counts += 1
                    continue

                print(f"Path count: {num_path}, Duration: {duration}")
                name_set.add(record["name1"])
                name_set.add(record["name2"])
                result_list.append(record)
                if len(result_list) == n:
                    break

        print(f"Retry counts: {retry_counts}")

        for line in result_list:
            question = q.format(line["name1"], line["name2"])
            q_entity = {
                "qid": len(questions),
                "question": question,
                "entity": {
                    line["name1"]: line["id1"],
                    line["name2"]: line["id2"],
                },
                "type": "multi_entity_concrete",
                "hops": n_hop,
                "answer": "N/A",
            }
            questions.append(q_entity)
            print(q_entity)

    with jsonlines.open(output_path, "w") as writer:
        for row in questions:
            writer.write(row)


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--path", type=str, default="../datasets/maple/Physics", required=True
)
argparser.add_argument(
    "--output-path", type=str, default="../benchmarks/hysics", required=True
)
args = argparser.parse_args()
print(args)

dataset_name = os.path.basename(args.path).lower()

# load the graph
graph = pickle.load(open(os.path.join(args.path, "graph.pkl"), "rb"))
print("NetworkX graph loaded")
print("# nodes:", graph.number_of_nodes())
print("# edges:", graph.number_of_edges())

# generate questions
gen_single_entity_abstract(
    graph,
    80,
    os.path.join(args.output_path, "single_entity_abstract.jsonl"),
)
gen_single_entity_concrete(
    graph,
    10,
    dataset_name,
    os.path.join(args.output_path, "single_entity_concrete.jsonl"),
)
gen_multi_entity_abstract(
    graph,
    20,
    [2, 3, 4, 5],
    os.path.join(args.output_path, "multi_entity_abstract.jsonl"),
)
gen_multi_entity_concrete(
    graph,
    20,
    dataset_name,
    os.path.join(args.output_path, "multi_entity_concrete.jsonl"),
)
