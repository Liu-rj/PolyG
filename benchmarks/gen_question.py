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


object_discovery_template = {
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
            MATCH (p:physics:paper {{id: '{}'}})
            -[:venue]->(v:physics:venue)
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


fact_check_template = {
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
        "Have authors '{}' and '{}' both collaborated with some other authors, not necessarily in the same work, and who are they?": {
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
    },
    "amazon": {
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
    },
}

nested_question_template = {
    "physics": {
        "Tell me about the academic contributions of the academic collaborators of the scholar '{}'.": {
            "cypher_template": """
        MATCH (author:physics:author)
        WITH author ORDER BY rand() LIMIT 100

        MATCH (author)-[:paper]->(:physics:paper)-[:author]->(collaborator:physics:author)
        WHERE author <> collaborator
        WITH author, count(DISTINCT collaborator) AS numCollaborators
        WHERE numCollaborators >= 5 AND numCollaborators <= 20
        RETURN author.name AS name1, author.id AS id1
        LIMIT 1
        """,
        },
        "Who are the academic collaborators of the author who writes both the paper '{}' and paper '{}'?": {
            "cypher_template": """
        MATCH (a:physics:author)
        WITH a ORDER BY rand() LIMIT 100

        MATCH (a)-[:paper]->(:physics:paper)-[:author]->(coAuthor:physics:author)
        WHERE coAuthor <> a
        WITH a, collect(DISTINCT coAuthor) AS collaborators
        WHERE size(collaborators) >= 5 AND size(collaborators) <= 20
        WITH a ORDER BY rand()

        MATCH (p1:physics:paper)-[:author]->(a), (p2:physics:paper)-[:author]->(a)
        WHERE p1 <> p2
        RETURN DISTINCT p1.name AS name1, p1.id AS id1, p2.name AS name2, p2.id AS id2
        LIMIT 1
        """,
            "cypher": """
        MATCH (p1:physics:paper {{id: '{}'}})-[:author]->(a:physics:author), (p2:physics:paper {{id: '{}'}})-[:author]->(a)
        MATCH (a)-[:paper]->(:physics:paper)-[:author]->(coAuthor:physics:author)
        WHERE coAuthor <> a
        RETURN DISTINCT coAuthor.name AS name
        """,
        },
        "What is the relationship between the scholar '{}' and the authors of the paper '{}'?": {
            "cypher_template": """
        MATCH (paper:physics:paper)
        WITH paper ORDER BY rand() LIMIT 1

        MATCH (paper)-[:author]->(author:physics:author)-[*3]->(anotherAuthor:physics:author)
        WHERE anotherAuthor <> author
        RETURN DISTINCT anotherAuthor.name AS name1, anotherAuthor.id AS id1, paper.name AS name2, paper.id AS id2
        LIMIT 1
        """,
        },
        "Have the scholars '{}' and '{}' both published work at the venue having the paper '{}'?": {
            "cypher_template": """
        MATCH (paper:physics:paper)-[:venue]->(venue:physics:venue)
        WITH DISTINCT venue, paper ORDER BY rand() LIMIT 1

        MATCH (author1:physics:author)
        -[:paper]->(paper1:physics:paper)
        -[:venue]->(venue:physics:venue)
        -[:paper]->(paper2:physics:paper)
        -[:author]->(author2:physics:author)
        WHERE paper1 <> paper2 AND author1 <> author2 AND NOT (author1)-[:paper]->(paper) AND NOT (author2)-[:paper]->(paper)
        RETURN DISTINCT author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2, paper.name AS name3, paper.id AS id3
        LIMIT 1
        """,
        },
    },
    "amazon": {
        "Give a broad introduction about the brands that are bought together with items of the brand '{}'?": {
            "cypher_template": """
        MATCH (i:amazon:brand)
        WITH i ORDER BY rand() LIMIT 100

        MATCH (i)-[:item]->(item:amazon:item)-[:bought_together_item]->(related:amazon:item)-[:brand]->(b:amazon:brand)-[:item]->(other:amazon:item)
        WITH i, count(DISTINCT b) AS numBrands, count(DISTINCT other) AS numItems
        WHERE numBrands >= 5 AND numBrands <= 20 AND numItems <= 100
        RETURN DISTINCT i.name AS name1, i.id AS id1
        LIMIT 1
        """,
        },
        "What are the brands that are commonly viewed with the brands '{}' and also bought together with the brand '{}'?": {
            "cypher_template": """
        MATCH (target:amazon:brand)
        WITH target ORDER BY rand() LIMIT 1

        MATCH (b1:amazon:brand)-[:item]->(:amazon:item)-[:also_viewed_item]->(:amazon:item)-[:brand]->(target:amazon:brand)
              -[:item]->(:amazon:item)-[:bought_together_item]->(:amazon:item)-[:brand]->(b2:amazon:brand)
        WHERE target <> b1 AND target <> b2 AND b1 <> b2
        WITH b1, b2, count(DISTINCT target) AS numBrands
        WHERE numBrands >= 1 AND numBrands <= 20
        RETURN DISTINCT b1.name AS name1, b1.id AS id1, b2.name AS name2, b2.id AS id2
        LIMIT 1
        """,
            "cypher": """
        MATCH (b1:amazon:brand {{id: '{}'}})-[:item]->(:amazon:item)-[:also_viewed_item]->(:amazon:item)-[:brand]->(target:amazon:brand)
              -[:item]->(:amazon:item)-[:bought_together_item]->(:amazon:item)-[:brand]->(b2:amazon:brand {{id: '{}'}})

        RETURN DISTINCT target.name AS name
        """,
        },
        "What is the relationship between the brand '{}' and the brand which has the item '{}'?": {
            "cypher_template": """
        MATCH (item:amazon:item)
        WITH item ORDER BY rand() LIMIT 1

        MATCH (item:amazon:item)-[:brand]->(brand1:amazon:brand)-[*3]->(brand2:amazon:brand)
        WHERE brand1 <> brand2
        RETURN DISTINCT brand2.name AS name1, brand2.id AS id1, item.name AS name2, item.id AS id2
        LIMIT 1
        """,
        },
        "Have the items of the brands '{}' and '{}' ever been also viewed with the items that are bought together with the item '{}'?": {
            "cypher_template": """
        MATCH (target:amazon:item)-[:bought_together_item]->(related:amazon:item)
        WITH target, related ORDER BY rand() LIMIT 1

        MATCH (related)<-[:also_viewed_item]-(i1:amazon:item)-[:brand]->(b1:amazon:brand),
              (related)<-[:also_viewed_item]-(i2:amazon:item)-[:brand]->(b2:amazon:brand)
        WHERE b1 <> b2
        RETURN DISTINCT b1.name AS name1, b1.id AS id1, b2.name AS name2, b2.id AS id2, target.name AS name3, target.id AS id3
        LIMIT 1
        """,
        },
    },
    "goodreads": {
        "Provide a comprehensive overview about the series whose books are similar to the publications of author '{}'.": {
            "cypher_template": """
        MATCH (author:goodreads:author)
        WITH author ORDER BY rand() LIMIT 100

        MATCH (author)-[:book]->(b:goodreads:book)-[:similar_books]->(:goodreads:book)-[:series]->(s:goodreads:series)
        WITH author, count(DISTINCT s) AS numSeries
        WHERE numSeries >= 5 AND numSeries <= 20
        RETURN DISTINCT author.name AS name1, author.id AS id1
        LIMIT 1
        """,
        },
        "What books are published by the collaborators of both the author '{}' and '{}'?": {
            "cypher_template": """
        MATCH (a:goodreads:author)
        WITH DISTINCT a ORDER BY rand() LIMIT 10

        MATCH (a)-[:book]->(book:goodreads:book)
        WITH a, collect(DISTINCT book) AS books
        WHERE size(books) >= 5 AND size(books) <= 20

        MATCH (a)-[:book]->(:goodreads:book)<-[:book]-(coauthor1:goodreads:author),
              (a)-[:book]->(:goodreads:book)<-[:book]-(coauthor2:goodreads:author)
        WHERE coauthor1 <> a AND coauthor2 <> a AND coauthor1 <> coauthor2
        RETURN DISTINCT coauthor1.name AS name1, coauthor1.id AS id1, coauthor2.name AS name2, coauthor2.id AS id2
        LIMIT 1
        """,
            "cypher": """
        MATCH (a1:goodreads:author {{id: '{}'}})-[:book]->(:goodreads:book)-[:author]->(target:goodreads:author),
              (a2:goodreads:author {{id: '{}'}})-[:book]->(:goodreads:book)-[:author]->(target)
        WHERE target <> a1 AND target <> a2
        MATCH (target)-[:book]->(book:goodreads:book)
        RETURN DISTINCT book.name AS name
        """,
        },
        "What is the relationship between the author '{}' and the author who has the book '{}'?": {
            "cypher_template": """
        MATCH (targetBook:goodreads:book)
        WITH targetBook ORDER BY rand() LIMIT 1

        MATCH (targetBook)-[:author]->(targetAuthor:goodreads:author)-[*3]->(otherAuthor:goodreads:author)
        WHERE otherAuthor <> targetAuthor
        RETURN DISTINCT otherAuthor.name AS name1, otherAuthor.id AS id1, targetBook.name AS name2, targetBook.id AS id2
        LIMIT 1
        """,
        },
        "Have the authors '{}' and '{}' ever published books in the same publisher that has the series '{}'?": {
            "cypher_template": """
        MATCH (p:goodreads:publisher)
        WITH p ORDER BY rand() LIMIT 1

        MATCH (p)-[:book]->(book1:goodreads:book)-[:author]->(a1:goodreads:author),
              (p)-[:book]->(book2:goodreads:book)-[:author]->(a2:goodreads:author)
        WHERE a1 <> a2 AND book1 <> book2
        MATCH (p)-[:book]->(book1)-[:series]->(s:goodreads:series)
        RETURN DISTINCT a1.name AS name1, a1.id AS id1, a2.name AS name2, a2.id AS id2, s.name AS name3, s.id AS id3
        LIMIT 1
        """
        },
    },
}


def gen_subject_centered(graph: nx.Graph, n: int, output_path: str):
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
        if graph.degree(node) < 10:  # type: ignore
            continue
        question = f"Give me a broad introduction about '{node_data['name']}'."
        q_entity = {
            "qid": i,
            "question": question,
            "entity": {node_data["name"]: node},
            "type": "<s,*,*>",
            "hops": 1,
            "answer": "N/A",
        }
        questions.append(q_entity)
        name_set.add(node_data["name"])
        print(q_entity)
        i += 1
    with jsonlines.open(output_path, "a") as writer:
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


def gen_object_discovery(n: int, graph_name: str, output_path: str):
    neo4j_url = neo4j_config["neo4j_url"]
    neo4j_auth = neo4j_config["neo4j_auth"]
    driver = GraphDatabase.driver(
        neo4j_url,
        auth=neo4j_auth,
        max_connection_pool_size=100,
        connection_timeout=60,
    )
    q_templates = object_discovery_template[graph_name]
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
                    "type": "<s,p,*>",
                    "hops": n_hop,
                    "answer": result_names,
                }
                questions.append(q_entity)
                selected_nodes.add(node_name)
                print(q_entity)
                i += 1

    with jsonlines.open(output_path, "a") as writer:
        for row in questions:
            writer.write(row)


def gen_predicate_discovery(
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
                "type": "<s,*,o>",
                "hops": n_hop,
                "answer": "N/A",
            }
            questions.append(q_entity)
            print(q_entity)
            i += 1

    with jsonlines.open(output_path, "a") as writer:
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


def gen_fact_check(n: int, graph_name: str, output_path: str):
    neo4j_url = neo4j_config["neo4j_url"]
    neo4j_auth = neo4j_config["neo4j_auth"]
    driver = GraphDatabase.driver(
        neo4j_url,
        auth=neo4j_auth,
        max_connection_pool_size=100,
        connection_timeout=60,
    )
    q_templates = fact_check_template[graph_name]
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
                "type": "<s,p,o>",
                "hops": n_hop,
                "answer": "N/A",
            }
            questions.append(q_entity)
            print(q_entity)

    with jsonlines.open(output_path, "a") as writer:
        for row in questions:
            writer.write(row)


def gen_nested_question(n: int, graph_name: str, output_path: str):
    neo4j_url = neo4j_config["neo4j_url"]
    neo4j_auth = neo4j_config["neo4j_auth"]
    driver = GraphDatabase.driver(
        neo4j_url,
        auth=neo4j_auth,
        max_connection_pool_size=100,
        connection_timeout=60,
    )
    q_templates = nested_question_template[graph_name]
    questions = []
    for it, (q, content) in enumerate(q_templates.items()):
        q_cypher_t = content["cypher_template"]
        q_cypher = content["cypher"] if "cypher" in content else None

        result_list = []
        answer_list = []
        with driver.session() as session:
            while len(result_list) < n:
                print(
                    f"Getting the next record for question: {q}, {len(result_list)}/{n}"
                )
                record = None
                try:
                    with session.begin_transaction(timeout=60) as tx:
                        results = tx.run(q_cypher_t)  # type: ignore
                        record = results.single()
                        if record is None:
                            print("No record found, retry")
                            continue
                except Exception as e:
                    print(f"Query failed: {e}")
                    continue

                values = []
                for key in record.keys():
                    if key.startswith("id"):
                        values.append(record[key])
                answer = "N/A"
                if q_cypher is not None:
                    answer = []
                    try:
                        print("Validating the cypher query")
                        with session.begin_transaction(timeout=30) as tx:
                            ret = tx.run(q_cypher.format(*values))  # type: ignore
                            for ele in ret:
                                if len(answer) > 20:
                                    raise Exception("Too many results")
                                answer.append(ele["name"])
                    except Exception as e:
                        print(f"Query failed: {e}")
                        continue

                if record is not None:
                    result_list.append(record)
                    answer_list.append(answer)

        for line, answer in zip(result_list, answer_list):
            if "name3" in line.keys():
                question = q.format(line["name1"], line["name2"], line["name3"])
                entity_mappings = {
                    line["name1"]: line["id1"],
                    line["name2"]: line["id2"],
                    line["name3"]: line["id3"],
                }
            elif "name2" in line.keys():
                question = q.format(line["name1"], line["name2"])
                entity_mappings = {
                    line["name1"]: line["id1"],
                    line["name2"]: line["id2"],
                }
            else:
                question = q.format(line["name1"])
                entity_mappings = {line["name1"]: line["id1"]}

            q_entity = {
                "qid": len(questions),
                "question": question,
                "entity": entity_mappings,
                "type": "nested",
                "answer": answer,
            }
            questions.append(q_entity)
            print(q_entity)

    with jsonlines.open(output_path, "a") as writer:
        for row in questions:
            writer.write(row)


argparser = argparse.ArgumentParser()
argparser.add_argument("--path", type=str, default="../datasets/physics", required=True)
argparser.add_argument(
    "--output-path", type=str, default="../benchmarks/physics", required=True
)
args = argparser.parse_args()
print(args)

dataset_name = os.path.basename(args.path.strip("/")).lower()

# load the graph
graph = pickle.load(open(os.path.join(args.path, "graph.pkl"), "rb"))
print("NetworkX graph loaded")
print("# nodes:", graph.number_of_nodes())
print("# edges:", graph.number_of_edges())


# generate questions
gen_subject_centered(
    graph,
    80,
    os.path.join(args.output_path, "subject_centered_raw.jsonl"),
)
gen_object_discovery(
    10,
    dataset_name,
    os.path.join(args.output_path, "object_discovery_raw.jsonl"),
)
gen_predicate_discovery(
    graph,
    20,
    [2, 3, 4, 5],
    os.path.join(args.output_path, "predicate_discovery_raw.jsonl"),
)
gen_fact_check(
    20,
    dataset_name,
    os.path.join(args.output_path, "fact_check_raw.jsonl"),
)
gen_nested_question(
    20,
    dataset_name,
    os.path.join(args.output_path, "nested_question_raw.jsonl"),
)
