from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, PromptHelper

documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex(
    documents=documents,
    prompt_helper=PromptHelper(
        max_input_size=5000,
        num_output=4096,  # converted to openai api davinci 003 max_tokens parameter (4096 is max)
        chunk_size_limit=2000,
        max_chunk_overlap=-68,  # adjust to num_output
    ))

input = "GPT indexをインストールする手順は？"
result = index.query(input)
print(result)
