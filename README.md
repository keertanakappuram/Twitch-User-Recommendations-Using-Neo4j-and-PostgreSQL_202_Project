# Twitch-User-Recommendations-Using-Neo4j-and-PostgreSQL_202_Project

Team Members-
Keertana Kappuram,
Shweta Nalluri,
Sreetama Chowdhury


https://github.com/user-attachments/assets/702429d8-9da4-43a1-9b4c-243274487af5

Process to run code:

1. Start by creating a new database called "twitch_recommendations" in PostgreSQL and then, create 'musae_ENGB_target.csv' (table called 'users' in PostgreSQL) and 'musae_ENGB_features.json' (table called 'user_features' in PostgreSQL) by running queries in 'postgresql_table_creation.sql' file.
2. Load data into the tables by running the 'load_json_data.py' code in Visual Code/preferred Python IDE. (change the database connection parameters according to your PostgreSQL credentials and make sure the datasets are in the same location as your python file)
3. Steps to be followed in Neo4j:
   a) Create a new project
   b) Create a Local DBMS for this project by clicking on the 'Add' dropdown next to the project name and set a name and password to your DBMS. Remember to use this same password while connecting to your Neo4j database through Python later in this project.
   c) Click on the Start button
   d) Hover your cursor on the DBMS name. Click on the three dots next to the Open button. In the given option, click on Open folder and then click on import. This will open a folder. Copy and paste the 'musae_ENGB_edges.csv', 'musae_ENGB_target.csv' and 'musae_ENGB_features.json' 
   e) Click on the name of your DBMS. This will open a right bar. You see three options Details, Plugins, Upgrade. Click on Plugins. Click on Graph Data Science Library and click on install.
   f) Hover your cursor on the DBMS name. Click on the Open button to open your Neo4j Browser.
   g) In the browser run the queries mentioned in the cypher file called 'neo4j.cypher'.
4. Make sure to import all dependencies mentioned in the 'requirments.txt' file.
5. Finally, run the 'scoring.py' file to get the final results. (Make sure to change the credentials required to connect to PostgreSQL and Neo4j in the python code.)
