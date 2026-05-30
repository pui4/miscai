import re
import json
try:
    import torch
    import chromadb
    import requests
    from transformers import AutoModel
    from chromadb import EmbeddingFunction, Embeddings, Documents
except:
    raise ImportError("The 'dlm' module is required to use this. Install it with 'pip install miscai[dlm]'.")

class DLM():
    def __init__(self, 
                 promt: str, 
                 model: str, 
                 convo_file: str,
                 api_key: str,
                 tools: tuple = (None, None), 
                 think: str = "high",
                 ) -> None:
        self.PROMT = promt
        self.MODEL = model
        self.API_KEY = api_key
        
        self.tools_def, self.tools_fn = tools
        self.think = think
        self.convo_file = convo_file

        self.jina_model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v5-text-small",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )
        self.jina_model = self.jina_model.to("cpu")

        chroma_client = chromadb.PersistentClient(path="./memory_db")
        self.collection = chroma_client.get_or_create_collection(
            name="ltm",
            embedding_function=self.JinaEmbeddingFunction(self.jina_model)
        )

        self.query_ef = self.JinaQueryEmbeddingFunction(self.jina_model)

        try:
            with open(self.convo_file, "r") as file:
                self.messages = json.load(file)
                self.msg_count = len(self.messages)
        except:
            self.messages = []
            self.msg_count = 0

    def save_memory(self, text: str, memory_id: str) -> None:
        self.collection.upsert(documents=[text], ids=[memory_id])

    def retrieve_memories(self, query: str, n: int = 3) -> str:
        query_embedding = self.query_ef([query])[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n)
        docs = results["documents"][0] # type: ignore
        return "\n".join(docs) if docs else ""
    
    def ask_LLM(self, text: str) -> str:
        system_prompt = self.PROMT

        memories = self.retrieve_memories(text)
        if memories:
            system_prompt += f"\n\nRelevant memories:\n{memories}"

        self.messages.append({"role": "user", "content": text})

        resp = requests.post(
            "https://api.inceptionlabs.ai/v1/chat/completions",
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.API_KEY}'
            },
            json={
                "model": self.MODEL,
                "messages": [{"role": "system", "content": system_prompt}] + self.messages[-10:],
                "reasoning_effort": self.think,
                "tools": self.tools_def
            }
        )
        resp_j = resp.json()
        print(f"API response: {resp_j}")

        reply = resp_j["choices"][0]["message"]["content"] or ""
        self.messages.append(self._serialize_assistant_message(resp_j["choices"][0]["message"]))
        print(f"\n{reply}\n")

        while resp_j["choices"][0]["message"]["tool_calls"]:
            result = self.call_tools(resp_j["choices"][0]["message"]["tool_calls"]) # type: ignore
            self.messages.extend(result)

            resp = requests.post(
                "https://api.inceptionlabs.ai/v1/chat/completions",
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.API_KEY}'
                },
                json={
                    "model": self.MODEL,
                    "messages": [{"role": "system", "content": system_prompt}] + self.messages[-10:],
                    "reasoning_effort": self.think,
                    "tools": self.tools_def
                }
            )

            resp_j = resp.json()
            print(f"API response: {resp_j}")
            reply = resp_j["choices"][0]["message"]["content"] or ""

            self.messages.append(self._serialize_assistant_message(resp_j["choices"][0]["message"]))
            print(f"\n{reply}\n")

        self.msg_count += 1
        self.save_memory(
            f"User: {text}\nAssistant: {reply}",
            memory_id=f"msg_{self.msg_count}"
        )

        with open(self.convo_file, "w") as file:
            json.dump(self.messages, file)

        return re.sub(r".*?</think>", "", reply, flags=re.DOTALL).strip()
    
    def call_tools(self, tool_calls: list) -> list:
        results = []

        for call in tool_calls:
            fn = call["function"]
            fn_name = fn["name"]
            fn_args = json.loads(fn["arguments"]) if fn["arguments"] else {}

            if fn_name in self.tools_fn:
                print(f"CALLING TOOL {fn_name}({fn_args})")
                try:
                    out = self.tools_fn[fn_name](**fn_args)
                    content = str(out) if out is not None else "Done."
                except Exception as e:
                    content = f"Error: {e}"
            else:
                content = f"Unknown tool: {fn_name}"

            results.append({
                "role": "tool",
                "content": content,
                "tool_call_id": call.get("id", fn_name)  # call is a dict, use .get()
            })

        return results

    def _serialize_assistant_message(self, message) -> dict:
        msg = {"role": "assistant", "content": message.get("content") or ""}
        tool_calls = message.get("tool_calls")
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.get("id", tc["function"]["name"]),
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]  # keep as raw string
                    }
                }
                for tc in tool_calls
            ]
        return msg

    class JinaEmbeddingFunction(EmbeddingFunction):
        def __init__(self, model) -> None:
            super().__init__()

            self.jina_model = model

        def __call__(self, input: Documents) -> Embeddings:
            embedings = self.jina_model.encode(
                texts=input,
                task="retrieval",
                prompt_name="document"
            )
            return embedings.tolist()

    class JinaQueryEmbeddingFunction(EmbeddingFunction):
        def __init__(self, model) -> None:
            super().__init__()

            self.jina_model = model

        def __call__(self, input: Documents) -> Embeddings:
            embeddings = self.jina_model.encode(
                texts=input,
                task="retrieval",
                prompt_name="query"
            )
            return embeddings.tolist()