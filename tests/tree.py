import loader

from core.datatypes.sequence import Sequence
from core.utils.timer import Timer

import services.client as client

@Timer.timed
def run(PASSES):
    top = client.HNClient.best()

    print(f"# Kids: {top.kids.length()}")

    curr = top
    descendants = Sequence([curr.kids])

    for i in range(PASSES):
        curr = client.HNClient.expand(curr.kids)
        descendants.append(
            curr.kids
        )

    print(f"{PASSES} Passes: {descendants.flatten().length()}")
    return top, descendants

d = run(3)
elapsed = Timer.times.first()
