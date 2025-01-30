from typing import Optional, Sequence


class Vocab:
    def __init__(
        self,
        tokens: Optional[list[str]] = None,
        stoi: Optional[dict[str, int]] = None,
    ):
        self.itos: list[str] = []  # index to string
        self.stoi: dict[str, int] = {}  # string to index
        self.default_index: int = -1

        if stoi:
            self.stoi = stoi.copy()
            self.itos = [""] * (max(stoi.values()) + 1)
            for token, index in stoi.items():
                self.itos[index] = token
        elif tokens:
            for token in tokens:
                self.append_token(token)

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.default_index)

    def __call__(self, tokens: Sequence[str]) -> list[int]:
        return self.lookup_indices(list(tokens))

    def set_default_index(self, index: int) -> None:
        self.default_index = index

    def get_default_index(self) -> Optional[int]:
        return self.default_index

    def insert_token(self, token: str, index: int) -> None:
        if index < 0 or index > len(self.itos):
            raise ValueError(f"Index {index} out of range [0, {len(self.itos)}]")
        if token in self.stoi:
            raise RuntimeError(f"Token '{token}' already exists in the vocab")

        self.itos.insert(index, token)
        self.stoi[token] = index
        for i in range(index + 1, len(self.itos)):
            self.stoi[self.itos[i]] = i

    def append_token(self, token: str) -> None:
        if token in self.stoi:
            raise RuntimeError(f"Token '{token}' already exists in the vocab")
        self.itos.append(token)
        self.stoi[token] = len(self.itos) - 1

    def lookup_token(self, index: int) -> str:
        if index < 0 or index >= len(self.itos):
            raise IndexError(f"Index {index} out of range [0, {len(self.itos)})")
        return self.itos[index]

    def lookup_tokens(self, indices: list[int]) -> list[str]:
        return [self.lookup_token(index) for index in indices]

    def lookup_indices(self, tokens: list[str]) -> list[int]:
        return [self[token] for token in tokens]

    def get_stoi(self) -> dict[str, int]:
        return self.stoi.copy()

    def get_itos(self) -> list[str]:
        return self.itos.copy()
