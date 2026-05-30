import os
import importlib.util

class ToolLoader():
    def __init__(self, tools_dir: str) -> None:
        self.tools_dir = tools_dir

    def get_tools(self) -> tuple[list, dict]:
        tools_list = []
        func_map = {}

        for filename in os.listdir(self.tools_dir):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue

            module_name = filename[:-3]
            filepath = os.path.join(self.tools_dir, filename)

            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module = importlib.util.module_from_spec(spec) # type: ignore
            spec.loader.exec_module(module) # type: ignore

            if not hasattr(module, "tool"):
                print(f"TOOL: {filename} DOESN'T HAVE TOOL DEFINITION!")
                continue

            tool_def = module.tool
            fn_name = tool_def["function"]["name"]

            if not hasattr(module, fn_name):
                print(f"TOOL {filename}'S FUNCTION DOESN'T MATCH DEFINITION!")
                continue

            tools_list.append(tool_def)
            func_map[fn_name] = getattr(module, fn_name)
            print(f"LOADED {filename} SUCCESSFULLY!")

        return tools_list, func_map