from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, PromptHelper

documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex(
    documents=documents,
    prompt_helper=PromptHelper(
        max_input_size=5000,  # LLM入力の最大トークン数
        num_output=256,  # LLM出力のトークン数
        chunk_size_limit=2000,  # チャンクのトークン数
        max_chunk_overlap=-10,  # チャンクオーバーラップの最大トークン数
    ))

input = "GPT indexをインストールする手順は？"
result = index.query(input)
print(result)
