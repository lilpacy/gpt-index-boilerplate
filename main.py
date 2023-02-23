from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex(documents)

input = "GPT indexをインストールする手順は？"
result = index.query(input)
print(result)
