from dataclasses import dataclass

class ContractError(ValueError):
    pass

@dataclass(frozen=True)
class Settings:
    model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    group_size: int = 8
    max_interactions: int = 10
    feedback_rows: int = 3
    train_temperature: float = 1.
    eval_temperature: float = 0.
    max_sequence_length: int = 4096

SETTINGS = Settings()

@dataclass(frozen=True)
class LengthContract:
    """Accounting is a caller decision. There is deliberately no default choice."""
    accounting: str
    provenance: str
    maximum: int = 4096

    def __post_init__(self) -> None:
        if self.maximum != SETTINGS.max_sequence_length:
            raise ContractError("Sequence length must remain 4096")
        if self.accounting not in {"total_context", "generated_and_feedback", "generated_only"}:
            raise ContractError("Explicit sequence accounting is required")
        if not self.provenance:
            raise ContractError("Sequence accounting needs provenance")

    def remaining(self, prompt: int, generated: int, feedback: int) -> int:
        used = generated
        if self.accounting != "generated_only":
            used += feedback
        if self.accounting == "total_context":
            used += prompt
        return max(0, self.maximum - used)
