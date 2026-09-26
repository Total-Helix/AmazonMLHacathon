import os
import pandas as pd

def create_mock_data():
    # Define directories
    dirs = ['dataset/train', 'dataset/test', 'output']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 1. Mock Source 1 Test Data (The source of truth)
    s1_data = {
        "entity_id": [f"S1-{str(i).zfill(5)}" for i in range(1, 11)],
        "business_name": ["Amazon", "Google", "Microsoft", "Apple", "Meta", "Tesla", "Netflix", "Spotify", "Uber", "Airbnb"],
        "business_address": ["410 Terry Ave", "1600 Amphitheatre", "1 Microsoft Way", "1 Apple Park", "1 Hacker Way", "1 Gigafactory", "100 Winchester", "1 Spotify St", "1455 Market St", "888 Brannan St"],
        "country": ["US", "US", "US", "US", "US", "US", "US", "US", "US", "US"]
    }
    pd.DataFrame(s1_data).to_csv('dataset/test/test_source1.tsv', sep='\t', index=False)

    # 2. Mock Source 2/3 Data (Candidates)
    s2_data = {
        "entity_id": [f"S2-{str(i).zfill(5)}" for i in range(1, 11)],
        "business_name": ["Amazon Corp", "Google LLC", "Micro Soft", "Apple Inc", "Facebook", "Tesla Motors", "Netflix Inc", "Spotify Tech", "Uber Tech", "Air Bnb"],
        "business_address": ["410 Terry Avenue", "1600 Amphitheatre Pkwy", "One Microsoft Way", "1 Apple Park Way", "1 Hacker Way", "1 Giga Factory", "100 Winchester Cir", "1 Spotify Street", "1455 Market", "888 Brannan"],
        "country": ["US"] * 10
    }
    pd.DataFrame(s2_data).to_csv('dataset/test/test_source2.tsv', sep='\t', index=False)

    s3_data = {
        "entity_id": [f"S3-{str(i).zfill(5)}" for i in range(1, 11)],
        "business_name": ["Amazon.com", "Alphabet", "MSFT", "Apple Computer", "Meta Platforms", "Tesla Inc", "Netflix Corp", "Spotify AB", "Uber Inc", "Airbnb Inc"],
        "business_address": ["410 Terry", "1600 Pkwy", "1 MS Way", "Apple Park", "Hacker Way", "Gigafactory", "Winchester", "Spotify St", "Market St", "Brannan St"],
        "country": ["US"] * 10
    }
    pd.DataFrame(s3_data).to_csv('dataset/test/test_source3.tsv', sep='\t', index=False)

    # 3. MOCK CANDIDATE PAIRS (What Role 2 is supposed to give you)
    # Give a few candidates for each S1 entity to train the ML model on
    candidate_pairs = {
        "source1_entity_id": s1_data["entity_id"],
        "candidate_entity_ids": [f"S2-{str(i).zfill(5)},S3-{str(i).zfill(5)}" for i in range(1, 11)]
    }
    pd.DataFrame(candidate_pairs).to_csv('output/candidate_pairs.tsv', sep='\t', index=False)

    # 4. MOCK MATCHING RESULTS (What Role 3 will eventually output)
    matching_results = {
        "source1_entity_id": s1_data["entity_id"],
        "matched_entity_ids": [f"S2-{str(i).zfill(5)}" for i in range(1, 11)]
    }
    pd.DataFrame(matching_results).to_csv('output/matching_results.tsv', sep='\t', index=False)

    print("Mock data generated successfully! Roles 3 and 4 can now start working.")

if __name__ == "__main__":
    create_mock_data()
