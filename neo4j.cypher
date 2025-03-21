//Loading the csv file containing the users to define the nodes and connect them with relationship
LOAD CSV WITH HEADERS FROM 'file:///musae_ENGB_edges.csv' AS row
MERGE (u1:User {new_id: toInteger(row.from)})
MERGE (u2:User {new_id: toInteger(row.to)})
CREATE (u1)-[:FOLLOWS]->(u2);

//Loading the csv file containing the features of the users and defining them to individual user
LOAD CSV WITH HEADERS FROM 'file:///musae_ENGB_target.csv' AS row
MERGE (u:User {new_id: toInteger(row.new_id)})
SET u.id = toInteger(row.id),
    u.days = toInteger(row.days),
    u.mature = toBoolean(row.mature),
    u.views = toInteger(row.views),
    u.partner = toBoolean(row.partner);

//Creating the graph projection using GDS library
CALL gds.graph.project(
'userGraph', // Name of the graph projection
'User', // Node label
{ FOLLOWS: { type: 'FOLLOWS', orientation: 'NATURAL' } } 
);


