from conceptnet_lite import download_db, connect
import os

db_path = os.path.join(os.path.dirname(__file__), 'eva_ai', 'knowledge_data', 'conceptnet.db')

print("=" * 60)
print("ConceptNet Database Downloader")
print("=" * 60)
print(f"Download will be saved to: {db_path}")
print(f"Expected size: ~1.85 GB")
print()
print("Starting download...")
print("(This may take 15-20 minutes depending on connection)")
print()

try:
    download_db()
    print()
    print("Download complete! Connecting to ConceptNet...")
    connect()
    print()
    print("=" * 60)
    print("SUCCESS: ConceptNet is now available!")
    print("=" * 60)

    # Test it
    from conceptnet_lite import edges_for, Label
    test_label = Label.get_or_create(label='human', language='en')
    edges = edges_for(test_label)
    print(f"Test query 'human' returned {len(list(edges))} edges")

except Exception as e:
    print()
    print("=" * 60)
    print(f"ERROR: {e}")
    print("=" * 60)
    print("You can retry by running this script again.")