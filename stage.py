from typing import Callable, Dict, Any, List, Tuple, Union, Optional

class Stage:
    """
    Encapsulates one pipeline step:
      - name: comma-separated output keys
      - model: model name for LLM stages (or None)
      - func: bound Assembler method implementing the stage
      - inputs: list of state keys to pull as args
      - beginning_instruction: optional system prompt injected before context
      - appended_instruction: optional system prompt injected after context
    """
    def __init__(
        self,
        name: str,
        model: Optional[str],
        func: Callable[..., Union[Any, Tuple[Any, ...]]],
        inputs: List[str],
        beginning_instruction: Optional[str] = None,
        appended_instruction: Optional[str] = None
    ):
        self.name                  = name
        self.model                 = model
        self.func                  = func
        self.inputs                = inputs
        self.beginning_instruction = beginning_instruction
        self.appended_instruction  = appended_instruction

    def execute(self, state: Dict[str, Any]) -> Union[Any, Tuple[Any, ...]]:
        args = [state[k] for k in self.inputs]
        # If an LLM is involved, pass through any system instructions
        if self.model:
            return self.func(
                model=self.model,
                *args,
                beginning_instruction=self.beginning_instruction,
                appended_instruction=self.appended_instruction
            )
        # Otherwise, run as before
        return self.func(*args)
