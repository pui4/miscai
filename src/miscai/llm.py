import re
import json
try:
    import torch
    import chromadb
    from transformers import AutoModel
    from chromadb import EmbeddingFunction, Embeddings, Documents
    from ollama import chat
    from ollama import ChatResponse
except:
    raise ImportError("The 'llm' module is required to use this. Install it with 'pip install miscai[llm]'.")

class LLM():
    def __init__(self, 
                 promt: str, 
                 model: str, 
                 convo_file: str,
                 tools: tuple = (None, None), 
                 think: bool = True,
                 ) -> None:
        self.PROMT = promt
        self.MODEL = model
        
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
    
    def _parse_arguments(self, arguments) -> dict:
        """Ensure tool call arguments are always a dict, never a JSON string."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, ValueError):
                return {}
        return dict(arguments) if arguments else {}

    def ask_LLM(self, text: str) -> str:
        system_prompt = self.PROMT

        memories = self.retrieve_memories(text)
        if memories:
            system_prompt += f"\n\nRelevant memories:\n{memories}"

        self.messages.append({"role": "user", "content": text})

        resp: ChatResponse = chat(
            model=self.MODEL,
            messages=[{"role": "system", "content": system_prompt}] + self.messages[-10:],
            stream=False,
            think=self.think,
            tools=self.tools_def
        )

        reply = resp.message.content or ""
        self.messages.append(self._serialize_assistant_message(resp.message))
        print(f"\n{reply}\n")

        while resp.message.tool_calls:
            result = self.call_tools(resp.message.tool_calls) # type: ignore
            self.messages.extend(result)

            resp: ChatResponse = chat(
                model=self.MODEL,
                messages=[{"role": "system", "content": system_prompt}] + self.messages[-10:],
                stream=False,
                think=self.think,
                tools=self.tools_def
            )

            reply = resp.message.content or ""
            self.messages.append(self._serialize_assistant_message(resp.message))
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
            fn = call.function
            fn_name = fn.name
            fn_args = self._parse_arguments(fn.arguments)

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
                "tool_call_id": getattr(call, "id", fn_name)
            })

        return results

    def _serialize_assistant_message(self, message) -> dict:
        msg = {"role": "assistant", "content": message.content or ""}
        tool_calls = message.tool_calls
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": getattr(tc, "id", tc.function.name),
                    "function": {
                        "name": tc.function.name,
                        # Always store as dict so reloaded messages stay valid
                        "arguments": self._parse_arguments(tc.function.arguments)
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