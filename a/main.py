from aco.aco import ACO
from time import time
import json

if __name__ == "__main__":
    tests = [
        {"pop": 10, "rho": 0.08, "alpha": 2, "beta": 75, "its": 700, "filepath": "entradas/entrada2.txt"},

    ]

    def start(col, i):
        res = col.start()
        print(i)
        return res

    results = []
    for i, test in enumerate(tests):
        s = time()
        col = ACO(**test)
        res = [start(col, i) for i in range(30)]
        e = time()
        results.append(test | {"time": e - s, "results": res})
        with open(f"tests/asdas.json", "w") as fp:
            json.dump(results, fp, indent=3)
        results = []
