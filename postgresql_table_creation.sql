CREATE TABLE user_features (
    new_id INTEGER PRIMARY KEY,
    features JSON NOT NULL
);

CREATE TABLE users (
    ids INTEGER,
    days INTEGER,
    mature BOOLEAN,
    views INTEGER,
    partner BOOLEAN,
    new_id INTEGER PRIMARY KEY
);