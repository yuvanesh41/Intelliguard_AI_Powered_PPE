CREATE TABLE violations_meta (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    anomaly_detected BOOLEAN NOT NULL,
    image_path TEXT
);

CREATE TABLE violation_details (
    id UUID REFERENCES violations_meta(id),
    label TEXT NOT NULL,
    confidence FLOAT NOT NULL
);