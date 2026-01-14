from typing import List


class DocStore():
    def __init__(self):
        self.store = dict()

    def add_document(self, text: str, source_name: str):
        next_key = None
        if len(self.store) == 0:
            next_key = 0
        else:
            next_key = list(self.store.keys())[-1] + 1

        self.store[next_key] = f"{source_name}: {text}\n"
        return next_key

    def get_document(self, doc_id: int):
        return self.store.get(doc_id)

    def get_documents(self, doc_ids: List[int]):
        results = list()
        for id in doc_ids:
            found = self.get_document(id)
            if found is None:
                continue
            else:
                results.append(found)
        return results
