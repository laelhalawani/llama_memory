
from gguf_llama import LlamaAI
import chromadb as vdb
import util_helper.file_handler as fh


class AIMemory:
    def __init__(self, llama_ai:LlamaAI, presistent: bool = True, vdb_path: str = "./memory/vdb/") -> None:
        if presistent:
            fh.create_dir(dir_path=vdb_path)
            self.vdb_memory = vdb.PersistentClient(path=vdb_path)
        else:
            self.vdb_memory = vdb.Client()
        self.collection = self.vdb_memory.get_or_create_collection("memory")
        self.last_memory_id = 0
        if llama_ai is not None:
            self.llm = llama_ai
        else:
            raise Exception("No language model or llm config provided!")
        self._memory_overlap_threshold: float = 0.3 #value between 0 and 1
    
    def batch_input(self, input_str: str, overlap_threshold: float = 0.3) -> list:
        input_str = input_str.strip()
        input_str_len = len(input_str)
        batched_input = []
        if self.llm.is_prompt_within_limit(input_str):
            batched_input.append(input_str)
        else:
            #split into batches
            batch = ""
            token_overlap = int(overlap_threshold * self.llm.max_tokens)
            token_non_overlap = self.llm.max_tokens - token_overlap

            for i in range(input_str_len):
                if self.llm.count_tokens(batch+input_str[i]) > token_non_overlap+int(token_overlap/2):
                    print(f"Got non-overlapping batch: {batch}")
                    print(f"Looking for overlaping part...")
                    for j in range(i, input_str_len):
                        if self.llm.count_tokens(batch+input_str[j]) > int(token_overlap/2):
                            break
                        else:
                            batch += input_str[j]
                    batched_input.append(batch)
                    batch = ""
                else:
                    batch += input_str[i]

        return batched_input

        
    def add_memory(self, memory_str: str, memory_metadata: dict = None, update_if_exists: bool = True) -> None:
        insert_method : callable = self.collection.add
        if update_if_exists:
            insert_method = self.collection.upsert

        if memory_metadata is not None:
            memory_metadata = [memory_metadata]

        batched_memories = self.batch_input(memory_str)
        if len(batched_memories) > 1:
            print(f"Had to split input into {len(batched_memories)} batches.")
        for memory_piece in batched_memories:
            embedded_memory_piece = self.llm.create_embeddings(memory_piece)
            id = [str(self.last_memory_id)]
            insert_method(
                embeddings=embedded_memory_piece,
                metadatas=memory_metadata,
                ids=id
                )
            self.last_memory_id += 1
    def add_memories(self, memory_strs: list, memory_metadata: list = None, update_if_exists: bool = True) -> None:
        for i in range(len(memory_strs)):
            if memory_metadata is not None:
                mem_mtd = memory_metadata[i]
            else: mem_mtd = None
            self.add_memory(memory_str=memory_strs[i], memory_metadata=mem_mtd, update_if_exists=update_if_exists)
    
    def get_memory(self, memory_id: int) -> list:
        return self.collection.get(ids=[str(memory_id)])

    def find_memories_as_text(self, query_str: str, n_results: int = 5, only_text: bool = False, include_metadata: bool = False, chronological: bool = False) -> list:
        found_memories = self.find_memories(query_str=query_str, n_results=n_results)
        preserved_information = []
        if only_text:
            text_memories_list = found_memories['texts']
            if include_metadata:
                metadatas_memories_list = found_memories['metadatas']
                zipped = zip(text_memories_list, metadatas_memories_list)
                for t, m in zipped:
                    combined_metadata: str = ""
                    for key in m:
                        combined_metadata += f"{m[key]}\n"
                    preserved_information.append(f"{combined_metadata}{t}")
            else:
                preserved_information = text_memories_list
        else:
            preserved_information = found_memories
        return preserved_information
    
    def find_memories(self, query_str: str, n_results: int = 5) -> list:
        #default sorting of db query api is by distance
        memories = self.collection.query(query_texts=[query_str], n_results=n_results)
        text_memories_list = memories['documents'][0] 
        metadatas_memories_list = memories['metadatas'][0] if memories['metadatas'] is not None else []
        ids_memories_list = memories['ids'][0] 
        distances_memories_list = memories['distances'][0]   
        embeddings_memories_list = memories['embeddings'][0] if memories['embeddings'] is not None else []
        longest_list_len = max(len(text_memories_list), len(metadatas_memories_list), len(ids_memories_list), len(distances_memories_list), len(embeddings_memories_list))
        #fill in emptuy data for zipping
        if len(metadatas_memories_list) < longest_list_len:
            metadatas_memories_list.extend([None] * (longest_list_len - len(metadatas_memories_list)))
        if len(embeddings_memories_list) < longest_list_len:
            embeddings_memories_list.extend([None] * (longest_list_len - len(embeddings_memories_list)))
        combined_memories = {
            "ids" : ids_memories_list,
            "distances" : distances_memories_list,
            "texts" : text_memories_list,
            "metadatas" : metadatas_memories_list,
            "embeddings" : embeddings_memories_list
        }
        return combined_memories

    def find_memories_closest(self, query_str: str, n_results: int = 5) -> list:
        memories = self.find_memories(query_str=query_str, n_results=n_results)
        ids_memories_list = memories['ids']
        text_memories_list = memories['texts']
        metadatas_memories_list = memories['metadatas']
        distances_memories_list = memories['distances']
        embeddings_memories_list = memories['embeddings']
        distances_memories_list = [float(distances_memories_list[i]) for i in range(len(distances_memories_list))]
        combined_memories = zip(
                     ids_memories_list,
                     distances_memories_list,
                     text_memories_list,
                     metadatas_memories_list,
                     embeddings_memories_list)
        combined_memories = sorted(combined_memories, key=lambda x: x[1])
        #split back up into separate lists
        ids_memories_list, distances_memories_list, text_memories_list, metadatas_memories_list, embeddings_memories_list = zip(*combined_memories)
        combined_memories = {
            "ids" : ids_memories_list,
            "distances" : distances_memories_list,
            "texts" : text_memories_list,
            "metadatas" : metadatas_memories_list,
            "embeddings" : embeddings_memories_list
        }
        print("Combined Memories By Similarity Distance")
        print(combined_memories)
        return combined_memories
        
    def find_memories_chronological(self, query_str: str, n_results: int = 5) -> list:
        ids_memories_list, distances_memories_list, text_memories_list, metadatas_memories_list, embeddings_memories_list = self.find_memories(query_str=query_str, n_results=n_results)
        #convert ids to int
        ids_memories_list = [int(i) for i in range(len(ids_memories_list))]
        #zip together
        combined_memories = zip(
                     ids_memories_list, 
                     distances_memories_list, 
                     text_memories_list, 
                     metadatas_memories_list, 
                     embeddings_memories_list)
        
        #sort by id
        combined_memories = sorted(combined_memories, key=lambda x: x[0])
        #split back up into separate lists
        ids_memories_list, distances_memories_list, text_memories_list, metadatas_memories_list, embeddings_memories_list = zip(*combined_memories)
        #create a dictionary
        combined_memories = {
            "ids" : ids_memories_list,
            "distances" : distances_memories_list,
            "texts" : text_memories_list,
            "metadatas" : metadatas_memories_list,
            "embeddings" : embeddings_memories_list
        }
        print("Combined Memories Chronological")
        print(combined_memories)
        return combined_memories
       





