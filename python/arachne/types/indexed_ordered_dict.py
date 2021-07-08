from collections import OrderedDict


class IndexedOrderedDict(OrderedDict):
    def get_by_index(self, index: int):
        return list(self.items())[index]
