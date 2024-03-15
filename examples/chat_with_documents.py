from g4l.local import DocumentRetriever

engine = DocumentRetriever(
    files=['einstein-albert.txt'], 
    embed_model='SmartComponents/bge-micro-v2', # https://huggingface.co/spaces/mteb/leaderboard
    verbose=True,
    #reset_storage = True,
)

retrieval_data = engine.retrieve('what inventions did he do')

for node_with_score in retrieval_data:
    node = node_with_score.node
    score = node_with_score.score

    # Access the text content
    text = node.text

    # Access the metadata
    metadata = node.metadata
    page_label = metadata['page_label']
    file_name = metadata['file_name']
    # ... access other metadata fields as needed

    # Print or process the extracted information
    print(f"Text: {text}")
    print(f"Score: {score}")
    print(f"Page Label: {page_label}")
    print(f"File Name: {file_name}")
    print("---")
    
print("---------------------------------")

retrieval_data = engine.retrieve_for_llm('what invenstions did he do')
print(retrieval_data)