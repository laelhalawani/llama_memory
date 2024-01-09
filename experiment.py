from gguf_llama import LlamaAI
from gguf_modeldb import ModelDB


mdb = ModelDB()
zephyr_gguf = mdb.find_model("zephyr", "q4_k_m").model_path()
dolphin_gguf = mdb.find_model("dolphin", "q4_k_m").model_path()

#NEED TO FINISH THIS UP IT WORKS WEIRDLY
def batch_input(llm:LlamaAI, input_str: str, overlap_threshold: float = 0.3) -> list:
    input_str = input_str.strip()
    batched_tokens = []
    if llm.is_prompt_within_limit(input_str):
        batched_tokens.append(input_str)
    else:
        tokenized_input = llm.tokenize(input_str)
        #split into batches
        token_overlap = int(overlap_threshold * llm.max_tokens)
        token_non_overlap = llm.max_tokens - token_overlap
        batch = []
        for i in range(len(tokenized_input)):
            if len(batch)+1 > token_non_overlap+int(token_overlap/2):
                batch.extend(tokenized_input[i:i+int(token_overlap/2)])
                batched_tokens.append(batch)
                batch = []
            else:
                batch.append(tokenized_input[i])
    batched_tokens = [llm.untokenize(b) for b in batched_tokens]
    return batched_tokens

            

a_brief_story = """
In the quiet town of Eldridge, nestled between rolling hills and dense forests, there lived an enigmatic old man named Oliver. He was known for his peculiar habit of collecting and cataloging forgotten memories. One day, a curious young girl named Emily stumbled upon his ancient bookstore, hidden away on a cobblestone street. Intrigued, she struck up a conversation with Oliver, who welcomed her into his world of forgotten tales.
As they explored dusty shelves filled with forgotten journals and faded photographs, Emily discovered the magic within each memory. Oliver recounted tales of lost love, childhood dreams, and moments that time had tried to erase. Together, they embarked on a journey to revive these memories, weaving a tapestry of stories that brought joy, sadness, and nostalgia to the once-sleepy town.
Word spread, and soon the bookstore became a sanctuary for those seeking solace in the past. Eldridge transformed into a haven for storytelling, where the echoes of forgotten memories breathed life into the present. In the heart of the town, Oliver and Emily's bookstore became a timeless refuge, a place where the past and present intertwined, leaving an indelible mark on the tapestry of Eldridge's history.
"""
llm_a = LlamaAI(model_gguf_path=dolphin_gguf, max_tokens=50, embedding=True)
batched = batch_input(llm_a, a_brief_story, 0.5)
print(len(batched))
for batch in batched:
    print(batch)
    print("\n")

